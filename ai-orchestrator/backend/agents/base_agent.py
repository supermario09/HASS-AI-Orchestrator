"""
Base agent class providing common functionality for all specialist agents.
"""
import os
import json
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import ollama
from ha_client import HAWebSocketClient
from mcp_server import MCPServer

# ---------------------------------------------------------------------------
# Global LLM semaphore — limits concurrent Ollama calls to 1.
#
# Without this, every agent fires its decision loop simultaneously and all
# of them call ollama.Client.chat() at the same time.  Ollama queues these
# but the combined latency causes each call to time out, which triggers the
# agent error path, which restarts the loop, which fires again — a feedback
# loop that destabilises the whole system.
#
# Serialising calls (semaphore=1) means agents queue politely: the second
# agent waits until the first one's LLM call returns before it starts.
# Decision intervals are long enough (≥120s by default) that the extra wait
# is imperceptible in practice.
# ---------------------------------------------------------------------------
_LLM_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Return (lazily creating) the process-wide LLM call semaphore.

    A single global semaphore ensures only one Ollama call runs at a time
    across all agents. This prevents memory pressure when multiple models
    would otherwise be loaded and inferring simultaneously on the same host.
    """
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        _LLM_SEMAPHORE = asyncio.Semaphore(1)
    return _LLM_SEMAPHORE


class BaseAgent(ABC):
    """
    Abstract base class for all specialist agents.
    Provides common functionality for LLM calls, decision logging, and tool execution.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        mcp_server: MCPServer,
        ha_client, #: Union[HAWebSocketClient, Callable[[], HAWebSocketClient]]
        skills_path: str,
        rag_manager: Optional[Any] = None,
        model_name: str = "deepseek-r1:8b",
        decision_interval: int = 300,
        broadcast_func: Optional[Any] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            mcp_server: MCP server for tool execution
            ha_client: Home Assistant WebSocket client
            skills_path: Path to SKILLS.md file
            rag_manager: Optional RAG Manager for context retrieval
            model_name: Ollama model name
            model_name: Ollama model name
            decision_interval: Seconds between decisions
            broadcast_func: Optional async callback for dashboard updates
        """
        self.agent_id = agent_id
        self.name = name
        self.mcp_server = mcp_server
        # Support lazy loading
        self._ha_provider = ha_client

        self.skills_path = Path(skills_path)
        self.rag_manager = rag_manager
        self.model_name = model_name
        self.decision_interval = decision_interval
        self.broadcast_func = broadcast_func
        self.status = "initializing"
        
        # Load skills from SKILLS.md
        self.skills = self.load_skills()
        
        # Ollama client — explicit timeout for remote Ollama (e.g. Mac Mini on LAN).
        # The default httpx read_timeout is 5s, which fires before a 1000-token
        # response can arrive over the network, causing silent empty responses.
        # 120s read gives the M4 plenty of headroom even for long reasonings.
        import httpx as _httpx
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.Client(
            host=ollama_host,
            timeout=_httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
        
        # Decision storage
        self.decision_dir = Path("/data/decisions") / agent_id
        self.decision_dir.mkdir(parents=True, exist_ok=True)
    
    def load_skills(self) -> Dict:
        """
        Load and parse SKILLS.md file.
        
        Returns:
            Dict containing parsed skill information
        """
        if not self.skills_path.exists():
            print(f"⚠️ SKILLS.md not found at {self.skills_path}, using defaults")
            return {
                "identity": f"{self.name} agent",
                "controllable_entities": [],
                "observable_entities": [],
                "tools": [],
                "decision_criteria": {},
                "performance_targets": {}
            }
        
        with open(self.skills_path, "r") as f:
            content = f.read()
        
        # Basic parsing (can be enhanced with proper markdown parser)
        skills = {
            "identity": self._extract_section(content, "Identity"),
            "controllable_entities": self._extract_list(content, "Controllable Entities"),
            "observable_entities": self._extract_list(content, "Observable Entities"),
            "tools": self._extract_section(content, "Available Tools"),
            "decision_criteria": self._extract_section(content, "Decision Criteria"),
            "performance_targets": self._extract_section(content, "Performance Targets"),
            "full_content": content
        }
        
        return skills
    
    def _extract_section(self, content: str, heading: str) -> str:
        """Extract content from a markdown section"""
        lines = content.split("\n")
        in_section = False
        section_lines = []
        
        for line in lines:
            if heading.lower() in line.lower() and line.startswith("#"):
                in_section = True
                continue
            if in_section:
                if line.startswith("#"):
                    break
                section_lines.append(line)
        
        return "\n".join(section_lines).strip()
    
    def _extract_list(self, content: str, heading: str) -> List[str]:
        """Extract list items from a markdown section"""
        section = self._extract_section(content, heading)
        items = []
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                items.append(line.lstrip("-*").strip().strip("`"))
        return items
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """
        Call Ollama LLM (non-blocking, serialised, with automatic retry).

        Tuned for a remote Ollama instance on a capable machine (e.g. M4 Mac Mini
        with 16GB RAM) accessed over a local network from the HA host.

        Performance tuning rationale:
        - keep_alive=-1: model stays loaded in Ollama memory indefinitely — avoids
          the 5-15s reload cost that fires after the default 5-min unload timeout.
          With 300s polling intervals this would hit every other decision cycle.
        - num_ctx=2048: decision prompts are 600-1200 tokens; 2048 halves the KV
          cache vs 4096, reducing prefill time and VRAM pressure.
        - temperature=0.3: lower entropy → more deterministic JSON, slightly faster
          sampling. High temperature adds noise without benefiting structured output.
        - num_predict=500: JSON decisions are 50-300 tokens; capping earlier stops
          runaway generation without truncating real responses.
        - repeat_penalty=1.0: disables the repeat-penalty computation pass, which
          is useless for short structured JSON output.

        Retries up to 3 times on empty response or transient network blips.

        Args:
            prompt:      Prompt text
            temperature: Sampling temperature (default 0.3 — deterministic JSON)
            max_tokens:  Maximum tokens to generate (default 1000 — must budget for think block)
        """
        import re

        _MAX_ATTEMPTS = 3
        _RETRY_DELAYS = [3, 8]   # short delays — network blips, not hardware slowness

        last_err: str = "unknown error"

        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                delay = _RETRY_DELAYS[attempt - 1]
                print(f"⚠️ {self.name} LLM attempt {attempt} empty/error — "
                      f"retrying in {delay}s… ({attempt}/{_MAX_ATTEMPTS - 1})")
                await asyncio.sleep(delay)

            try:
                async with _get_llm_semaphore():
                    response = await asyncio.to_thread(
                        self.ollama_client.chat,
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            # num_ctx = total context (prompt + output).  Decision prompts
                            # are 1000-1500 tokens; with num_ctx=2048 only ~500-1000 tokens
                            # remain for output.  deepseek-r1:8b's <think> block alone can
                            # exceed that, leaving zero tokens for the actual JSON response.
                            # 4096 gives ~2500-3000 tokens of output budget — enough for a
                            # full think block plus the JSON decision.
                            "num_ctx": 4096,
                            "think": False,
                            "repeat_penalty": 1.0, # disable penalty pass — wasteful for JSON
                        },
                        keep_alive=-1,             # never unload; avoids 5-15s reload cost
                        stream=False,
                    )

                content = response["message"]["content"]

                # Strip any remaining <think>...</think> blocks (Issue #12 failsafe)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                content = content.strip()

                if not content:
                    last_err = "LLM returned empty response"
                    continue   # retry

                return content

            except Exception as e:
                last_err = str(e)
                print(f"❌ {self.name} LLM attempt {attempt + 1} failed: {e}")

        print(f"❌ {self.name} LLM failed after {_MAX_ATTEMPTS} attempts: {last_err}")
        return f"ERROR: {last_err}"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from SKILLS.md"""
        prompt = f"""You are {self.skills['identity']}.

Your role is to make intelligent decisions about home automation based on current conditions.

Available Tools:
{self.skills['tools']}

Decision Criteria:
{self.skills['decision_criteria']}
"""
        # Inject RAG context if available in skills (populated during decide)
        if "relevant_knowledge" in self.skills:
             prompt += f"\nRELEVANT KNOWLEDGE (from memory/docs):\n{self.skills['relevant_knowledge']}\n"
             
        # Append output format instructions
        prompt += """
Respond with a JSON object containing your decision in this format:
{
  "reasoning": "Brief explanation of why you made this decision",
  "actions": [
    {
      "tool": "tool_name",
      "parameters": {
        "param1": "value1"
      }
    }
  ]
}

If no action is needed, return an empty actions array.
"""
        return prompt
    
    @abstractmethod
    async def decide(self, context: Dict) -> Dict:
        """
        Make a decision based on current context.
        Must be implemented by specialist agents.
        
        Args:
            context: Current state and context information
        
        Returns:
            Decision dict with reasoning and actions
        """
        pass
    
    async def execute(self, decision: Dict) -> List[Dict]:
        """
        Execute decision actions using MCP tools.
        
        Args:
            decision: Decision dict from decide()
        
        Returns:
            List of execution results
        """
        actions = decision.get("actions", [])
        results = []
        
        for action in actions:
            tool_name = action["tool"]
            parameters = action["parameters"]
            
            result = await self.mcp_server.execute_tool(
                tool_name=tool_name,
                parameters=parameters,
                agent_id=self.agent_id
            )
            
            results.append({
                "tool": tool_name,
                "parameters": parameters,
                "result": result
            })
        
        return results
    
    def log_decision(self, context: Dict, decision: Dict, results: List[Dict]):
        """Save decision to log file"""
        timestamp = datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "agent_id": self.agent_id,
            "context": context,
            "decision": decision,
            "execution_results": results,
            "dry_run": self.mcp_server.dry_run
        }
        
        log_file = self.decision_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

    async def retrieve_context(self, state_text: str) -> str:
        """
        Retrieve relevant context from RAG based on current state description.
        
        Args:
            state_text: Description of current situation to query against
            
        Returns:
            Formatted string of relevant knowledge
        """
        if not self.rag_manager:
            return ""
            
        results = self.rag_manager.query(
            query_text=state_text,
            collection_names=["knowledge_base", "entity_registry", "memory"],
            n_results=2
        )
        
        if not results:
            return ""
            
        knowledge_str = ""
        for res in results:
            source = res.get("source", "unknown")
            content = res.get("content", "").strip()
            knowledge_str += f"- [{source}] {content}\n"
            
        return knowledge_str
    
    def get_last_decision_file(self) -> Optional[Path]:
        """Get path to most recent decision log"""
        decision_files = sorted(self.decision_dir.glob("*.json"), reverse=True)
        return decision_files[0] if decision_files else None
    
    async def _broadcast_status(self, status: str):
        """Broadcast status update to dashboard"""
        if self.broadcast_func:
            await self.broadcast_func({
                "type": "agent_update",
                "data": {
                    "agent_id": self.agent_id,
                    "name": self.name,
                    "status": status,
                    "last_active": datetime.now().astimezone().isoformat()
                }
            })

    async def run_decision_loop(self):
        """Main decision loop that runs continuously"""
        self.status = "idle"
        # Wait until HA is connected before making the first decision.
        # Avoids "Home Assistant not connected" errors on every agent at startup.
        print(f"⏳ {self.name} waiting for Home Assistant connection...")
        for _ in range(60):  # wait up to 60 seconds
            ha = self.ha_client
            if ha and ha.connected:
                break
            await asyncio.sleep(1)
        print(f"✓ {self.name} HA connected — warming up {self.model_name}...")
        # Pre-warm: fire a minimal prompt so the model is loaded into Ollama's
        # GPU/RAM before the first real decision.  Without this, the first call
        # pays the model-load penalty (~5-15s) on top of generation time.
        try:
            # Use a small but realistic budget — enough for a think block + short reply.
            # "ping" with max_tokens=3 failed: deepseek-r1 generates <think> first even
            # when think:False is set, consuming the entire budget before any output.
            await self._call_llm(
                'Reply with the word "ready" and nothing else.',
                max_tokens=200,
                temperature=0.1,
            )
            print(f"✓ {self.name} model warm — decision loop started (interval: {self.decision_interval}s)")
        except Exception as e:
            print(f"⚠️ {self.name} warmup failed (non-fatal): {e}")
            print(f"✓ {self.name} decision loop started (interval: {self.decision_interval}s)")

        while True:
            try:
                self.status = "deciding"
                await self._broadcast_status("deciding")

                # Make decision
                context = await self.gather_context()
                decision = await self.decide(context)

                # Execute decision
                results = await self.execute(decision)

                # Log decision
                self.log_decision(context, decision, results)

                # Broadcast decision result
                if self.broadcast_func:
                    await self.broadcast_func({
                        "type": "decision",
                        "data": {
                            "timestamp": datetime.now().astimezone().isoformat(),
                            "agent_id": self.agent_id,
                            "reasoning": decision.get("reasoning", ""),
                            "action": str(decision.get("actions", [])),
                            "dry_run": self.mcp_server.dry_run
                        }
                    })

                self.status = "idle"
                await self._broadcast_status("idle")
                print(f"✓ {self.name} decision completed (waiting {self.decision_interval}s)")

                # Sleep at END of loop
                await asyncio.sleep(self.decision_interval)

            except Exception as e:
                self.status = "error"
                print(f"❌ {self.name} decision loop error: {e}")
                await self._broadcast_status("error")
                await asyncio.sleep(10)  # Back off on error
    
    @abstractmethod
    async def gather_context(self) -> Dict:
        """
        Gather current context for decision making.
        Must be implemented by specialist agents.
        
        Returns:
            Context dict with current state
        """
        pass

    @property
    def ha_client(self):
        """Lazy retrieval of HA client"""
        if callable(self._ha_provider):
            return self._ha_provider()
        return self._ha_provider

    @ha_client.setter
    def ha_client(self, value):
        self._ha_provider = value
