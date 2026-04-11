"""
Universal Agent — driven entirely by natural language instructions and a YAML entity list.

Stability improvements over upstream:
- Entity discovery result is cached for 5 minutes (reduces load on large HA instances)
- Heuristic fallback capped at 20 entities (keeps LLM context small)
- Domain filtering based on instruction keywords before loading all states
- Fixed: truncated _get_state_description() that silently returned None for static entities
"""

import asyncio
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from mcp_server import MCPServer
from ha_client import HAWebSocketClient

logger = logging.getLogger(__name__)

# Domains that can be controlled (used for heuristic filtering)
_CONTROL_DOMAINS = ["climate", "light", "switch", "lock", "cover", "fan", "media_player"]
_SENSOR_DOMAINS  = ["sensor", "binary_sensor"]
_ALL_INTERESTING = _CONTROL_DOMAINS + _SENSOR_DOMAINS

# Keyword → domain mappings for instruction-based pre-filtering
_KEYWORD_TO_DOMAINS: Dict[str, List[str]] = {
    "temperature": ["climate", "sensor"],
    "heat":        ["climate"],
    "cool":        ["climate"],
    "hvac":        ["climate"],
    "light":       ["light"],
    "bright":      ["light"],
    "lamp":        ["light"],
    "dim":         ["light"],
    "lock":        ["lock"],
    "door":        ["lock", "binary_sensor"],
    "window":      ["cover", "binary_sensor"],
    "blind":       ["cover"],
    "shutter":     ["cover"],
    "fan":         ["fan"],
    "music":       ["media_player"],
    "speaker":     ["media_player"],
    "motion":      ["binary_sensor"],
    "occupancy":   ["binary_sensor"],
    "security":    ["lock", "alarm_control_panel", "binary_sensor", "camera"],
    "alarm":       ["alarm_control_panel"],
    "camera":      ["camera"],
    "presence":    ["binary_sensor", "device_tracker"],
    "energy":      ["sensor", "switch"],
    "power":       ["sensor", "switch"],
    "switch":      ["switch"],
    "voice":       [],   # voice agent — no entity filter
    "announce":    [],
}

# Cache TTL for entity discovery (5 minutes)
_ENTITY_CACHE_TTL = timedelta(minutes=5)


def _infer_domains_from_instruction(instruction: str) -> List[str]:
    """Return a de-duped list of HA domains relevant to the instruction text."""
    lowered = instruction.lower()
    domains: List[str] = []
    for kw, doms in _KEYWORD_TO_DOMAINS.items():
        if kw in lowered:
            domains.extend(doms)
    return list(dict.fromkeys(domains))  # preserve order, deduplicate


class UniversalAgent(BaseAgent):
    """
    A universal agent that operates based on natural language instructions
    and a dynamic list of entities, rather than hardcoded logic.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        instruction: str,
        mcp_server: MCPServer,
        ha_client: HAWebSocketClient,
        entities: List[str],
        rag_manager: Optional[Any] = None,
        model_name: str = "mistral:7b-instruct",
        decision_interval: int = 120,
        broadcast_func: Optional[Any] = None,
        knowledge: str = ""
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            mcp_server=mcp_server,
            ha_client=ha_client,
            skills_path="UNIVERSAL_AGENT",
            rag_manager=rag_manager,
            model_name=model_name,
            decision_interval=decision_interval,
            broadcast_func=broadcast_func,
        )
        self.instruction = instruction
        self.entities = entities
        self.knowledge = knowledge

        # Entity discovery cache
        self._discovered_entities: List[str] = []
        self._entity_cache_expires: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Skills prompt
    # ------------------------------------------------------------------

    def _load_skills(self) -> str:
        return f"""
# AGENT ROLE: {self.name}
# TARGET ENTITIES: {', '.join(self.entities) if self.entities else 'Dynamic/All'}

# PRIMARY INSTRUCTION
{self.instruction}

# KNOWLEDGE / CONTEXT
{self.knowledge if self.knowledge else "No additional context provided."}

# CAPABILITIES & SAFETY
1. You have access to Home Assistant services via the 'call_ha_service' tool.
2. ACCESS RESTRICTIONS: You CANNOT access 'shell_command', 'hassio', 'script', or 'automation' domains.
3. APPROVAL REQUIRED: High-impact actions (e.g., unlocking doors, disarming alarms) will be queued for human approval.
4. VALIDATION: Generic service calls (e.g. set_temperature) must still adhere to safety limits (10-30°C).
"""

    # ------------------------------------------------------------------
    # Entity discovery (with caching)
    # ------------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        return (
            bool(self._discovered_entities)
            and self._entity_cache_expires is not None
            and datetime.now() < self._entity_cache_expires
        )

    def _update_cache(self, entities: List[str]):
        self._discovered_entities = entities
        self._entity_cache_expires = datetime.now() + _ENTITY_CACHE_TTL

    async def _get_state_description(self) -> str:
        """
        Build a state description for the LLM.

        If self.entities is populated (from agents.yaml), fetch only those.
        If empty (dynamic mode), use RAG or domain-filtered heuristic discovery,
        with results cached for 5 minutes to avoid hammering HA on every cycle.
        """
        states: List[str] = []

        # ---- Static entity list ----------------------------------------
        if self.entities:
            states.append("Configured Entity States:")
            for eid in self.entities:
                try:
                    s = await self.ha_client.get_states(eid)
                    if s:
                        friendly = s.get("attributes", {}).get("friendly_name", eid)
                        states.append(f"- {friendly} ({eid}): {s['state']}")
                except Exception as e:
                    states.append(f"- {eid}: [error fetching state: {e}]")
            states.append(f"- Time: {datetime.now().strftime('%H:%M')}")
            return "\n".join(states)

        # ---- Dynamic discovery -----------------------------------------
        try:
            # 1. Serve from cache if still fresh
            if self._is_cache_valid():
                states.append("Cached Entity Discovery:")
                for eid in self._discovered_entities:
                    try:
                        s = await self.ha_client.get_states(eid)
                        if s:
                            friendly = s.get("attributes", {}).get("friendly_name", eid)
                            states.append(f"- {friendly} ({eid}): {s['state']}")
                    except Exception:
                        pass
                states.append(f"- Time: {datetime.now().strftime('%H:%M')}")
                return "\n".join(states)

            # 2. Try RAG semantic search
            if self.rag_manager:
                try:
                    logger.info(f"🔍 Semantic entity search: '{self.instruction[:60]}...'")
                    loop = asyncio.get_running_loop()

                    def _run_rag():
                        return self.rag_manager.query(
                            query_text=self.instruction,
                            collection_names=["entity_registry"],
                            n_results=10,
                        )

                    rag_results = await loop.run_in_executor(None, _run_rag)

                    found_entities: List[str] = []
                    for res in (rag_results or []):
                        content = res.get("content", "")
                        if "nomic-embed-text" in content or "error" in content.lower():
                            continue
                        match = re.search(r"Entity: ([a-z0-9_]+\.[a-z0-9_]+)", content)
                        if match:
                            found_entities.append(match.group(1))

                    if found_entities:
                        self._update_cache(found_entities)
                        states.append("Semantic Entity Discovery:")
                        for eid in found_entities:
                            try:
                                s = await self.ha_client.get_states(eid)
                                if s:
                                    friendly = s.get("attributes", {}).get("friendly_name", eid)
                                    states.append(f"- {friendly} ({eid}): {s['state']}")
                            except Exception:
                                pass
                        states.append(f"- Time: {datetime.now().strftime('%H:%M')}")
                        return "\n".join(states)

                except Exception as rag_err:
                    logger.warning(f"⚠️ Semantic search failed, falling back to heuristic: {rag_err}")

            # 3. Heuristic fallback — domain-filtered, capped at 20 entities
            #    Infer relevant domains from instruction keywords to avoid loading
            #    all 2600+ entities on every cycle.
            target_domains = _infer_domains_from_instruction(self.instruction) or _ALL_INTERESTING

            all_states = await self.ha_client.get_states()

            # Filter by domain, sort controllable first
            def _priority(s):
                dom = s["entity_id"].split(".")[0]
                return (0 if dom in _CONTROL_DOMAINS else 1, s["entity_id"])

            filtered = sorted(
                [s for s in all_states if s["entity_id"].split(".")[0] in target_domains],
                key=_priority,
            )[:20]   # cap at 20 — keeps LLM context small

            discovered_ids = [s["entity_id"] for s in filtered]
            self._update_cache(discovered_ids)

            states.append(f"Heuristic Entity Discovery (domains: {', '.join(target_domains)}):")
            for s in filtered:
                eid = s["entity_id"]
                friendly = s.get("attributes", {}).get("friendly_name", eid)
                states.append(f"- {friendly} ({eid}): {s['state']}")

            states.append(f"- Time: {datetime.now().strftime('%H:%M')}")
            return "\n".join(states)

        except Exception as e:
            if self.ha_client and not self.ha_client.connected:
                logger.warning("⚠️ Entity discovery paused — HA client disconnected")
            else:
                logger.error(f"❌ Entity discovery fatal error: {e}")
            return "Error: Could not discover entities. Please check Home Assistant connection."

    # ------------------------------------------------------------------
    # Context gathering
    # ------------------------------------------------------------------

    async def gather_context(self) -> Dict:
        state_desc = await self._get_state_description()
        return {
            "timestamp": datetime.now().isoformat(),
            "state_description": state_desc,
            "instruction": self.instruction,
        }

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    async def decide(self, context: Dict) -> Dict:
        """Make decision based on instruction and current state."""

        # Discover relevant services for the current entity set
        relevant_services_text = ""
        try:
            active_entities = self.entities or self._discovered_entities
            if active_entities:
                domains = set(e.split(".")[0] for e in active_entities)
                all_services = await self.ha_client.get_services()
                lines = []
                for domain in domains:
                    if domain in all_services:
                        lines.append(f"- {domain}: {', '.join(all_services[domain].keys())}")
                if lines:
                    relevant_services_text = "\nAVAILABLE HA SERVICES (Use EXACT names):\n" + "\n".join(lines)
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch services: {e}")

        state_desc = context.get("state_description", "No state data")

        prompt = f"""
{self._load_skills()}

CURRENT SITUATION:
Time: {context['timestamp']}

ENTITY STATES:
{state_desc}

{relevant_services_text}

CRITICAL RULES:
1. You MUST ONLY use entity IDs listed in 'ENTITY STATES'. Do NOT guess or hallucinate IDs.
2. If the entity you need is not listed, use the 'log' tool to report "Entity X not found".
3. Use 'call_ha_service' only for generic services. For climate/lights, prefer specialized tools.
4. Respond with VALID STANDARD JSON only. NO COMMENTS (// or /*) inside JSON.
5. Do not wrap in markdown blocks.

TOOL USAGE EXAMPLES:
- Specific: {{"tool": "set_temperature", "parameters": {{"entity_id": "climate.lounge", "temperature": 21.0}}}}
- Generic: {{"tool": "call_ha_service", "parameters": {{"domain": "light", "service": "turn_on", "entity_id": "light.living_room", "service_data": {{"brightness_pct": 50}}}}}}
- Voice:   {{"tool": "speak_tts", "parameters": {{"message": "Good morning!", "media_player": "media_player.kitchen_display"}}}}

Based on your PRIMARY INSTRUCTION and CURRENT SITUATION, determine if any action is needed.
Respond with a JSON object containing 'reasoning' and 'actions'.
Each action MUST have a 'tool' field and 'parameters'.
"""
        response = await self._call_llm(prompt)

        if response.startswith("ERROR:"):
            return {"reasoning": f"LLM failure: {response[6:].strip()}", "actions": []}

        if not response.strip():
            return {"reasoning": "LLM returned empty response.", "actions": []}

        try:
            clean = response.strip()
            # Strip markdown code fences if present
            clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
            clean = re.sub(r"\s*```$", "", clean)

            def _loose_json(text: str):
                text = re.sub(r"//.*", "", text)
                text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    text = re.sub(r",\s*}", "}", text)
                    text = re.sub(r",\s*]", "]", text)
                    return json.loads(text)

            data = _loose_json(clean)

            valid_actions = []
            for action in data.get("actions", []):
                if "tool" in action:
                    valid_actions.append(action)
                elif "service" in action:
                    valid_actions.append({"tool": "call_ha_service", "parameters": action})

            data["actions"] = valid_actions
            return data

        except Exception as e:
            logger.error(f"❌ JSON parse error: {e}\nRaw: {response[:200]}")
            return {"reasoning": f"Failed to parse LLM response: {e}", "actions": []}
