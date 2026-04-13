"""
Tests for v1.2.2 LLM efficiency improvements.

Covers:
1.  keep_alive=-1 passed to ollama_client.chat() in _call_llm
2.  num_ctx is 2048 (down from 4096) in _call_llm options
3.  Default temperature is 0.3 (down from 0.7) in _call_llm signature
4.  Default max_tokens is 500 (down from 1000) in _call_llm signature
5.  repeat_penalty=1.0 in _call_llm options
6.  VisionAgent Ollama path uses keep_alive=-1
7.  VisionAgent Ollama path uses num_ctx=2048
8.  VisionAgent Ollama path uses repeat_penalty=1.0
9.  run_decision_loop fires a warmup _call_llm before the main while loop
10. UniversalAgent caches get_services() — second call does NOT hit ha_client
11. UniversalAgent services cache expires after 10 minutes (re-fetches after)
12. UniversalAgent services cache initialises as None (lazy)
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch
import pytest

# ── path + heavy-dep stubs ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

for _mod in (
    "langgraph", "langgraph.graph", "chromadb", "chromadb.config",
    "chromadb.utils", "langgraph.checkpoint", "langgraph.checkpoint.memory",
    "fastapi", "fastapi.routing", "google.generativeai", "PIL", "PIL.Image",
):
    sys.modules.setdefault(_mod, MagicMock())

sys.modules["langgraph.graph"].END = "END"
sys.modules["langgraph.graph"].StateGraph = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_base_agent(model_name="deepseek-r1:8b"):
    from agents.base_agent import BaseAgent

    class _Agent(BaseAgent):
        async def decide(self, ctx):
            return {"actions": []}
        async def gather_context(self):
            return {}

    mcp = MagicMock()
    mcp.dry_run = False

    with patch("agents.base_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
        agent = _Agent(
            agent_id="t", name="T",
            mcp_server=mcp, ha_client=MagicMock(),
            skills_path="/nonexistent/SKILLS.md",
            model_name=model_name,
        )
    return agent


def _make_universal_agent(entities=None):
    from agents.universal_agent import UniversalAgent
    mcp = MagicMock()
    mcp.dry_run = False
    with patch("agents.base_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
        return UniversalAgent(
            agent_id="ua", name="UA",
            instruction="manage lights",
            mcp_server=mcp, ha_client=MagicMock(),
            entities=entities or ["light.hall"],
        )


async def _fake_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 1-5. _call_llm options
# ─────────────────────────────────────────────────────────────────────────────

class TestCallLlmOptions:

    @pytest.mark.asyncio
    async def test_keep_alive_minus_one_passed_to_chat(self):
        """_call_llm must pass keep_alive=-1 to ollama_client.chat()."""
        agent = _make_base_agent()
        chat_kwargs = {}

        def _capture_chat(*args, **kwargs):
            chat_kwargs.update(kwargs)
            return {"message": {"content": "ok"}}

        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.side_effect = _capture_chat

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.base_agent._LLM_SEMAPHORE", asyncio.Semaphore(1)):
            await agent._call_llm("test")

        assert "keep_alive" in chat_kwargs, "keep_alive not passed to ollama_client.chat()"
        assert chat_kwargs["keep_alive"] == -1, (
            f"keep_alive={chat_kwargs['keep_alive']!r}, expected -1"
        )

    @pytest.mark.asyncio
    async def test_num_ctx_is_2048(self):
        """_call_llm options must include num_ctx=2048."""
        agent = _make_base_agent()
        captured_options = {}

        def _capture(model, messages, options, **kwargs):
            captured_options.update(options)
            return {"message": {"content": "ok"}}

        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.side_effect = _capture

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.base_agent._LLM_SEMAPHORE", asyncio.Semaphore(1)):
            await agent._call_llm("test")

        assert captured_options.get("num_ctx") == 2048, (
            f"num_ctx={captured_options.get('num_ctx')}, expected 2048"
        )

    def test_default_temperature_is_0_3(self):
        """_call_llm default temperature must be 0.3."""
        import inspect
        from agents.base_agent import BaseAgent
        sig = inspect.signature(BaseAgent._call_llm)
        default = sig.parameters["temperature"].default
        assert default == 0.3, f"Default temperature is {default}, expected 0.3"

    def test_default_max_tokens_is_500(self):
        """_call_llm default max_tokens must be 500."""
        import inspect
        from agents.base_agent import BaseAgent
        sig = inspect.signature(BaseAgent._call_llm)
        default = sig.parameters["max_tokens"].default
        assert default == 500, f"Default max_tokens is {default}, expected 500"

    @pytest.mark.asyncio
    async def test_repeat_penalty_is_1_0(self):
        """_call_llm options must include repeat_penalty=1.0."""
        agent = _make_base_agent()
        captured_options = {}

        def _capture(model, messages, options, **kwargs):
            captured_options.update(options)
            return {"message": {"content": "ok"}}

        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.side_effect = _capture

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.base_agent._LLM_SEMAPHORE", asyncio.Semaphore(1)):
            await agent._call_llm("test")

        assert captured_options.get("repeat_penalty") == 1.0, (
            f"repeat_penalty={captured_options.get('repeat_penalty')}, expected 1.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6-8. VisionAgent Ollama path options
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionAgentOllamaOptions:

    def _make_vision_agent(self):
        with patch("agents.vision_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
            from agents.vision_agent import VisionAgent
            return VisionAgent(
                agent_id="v", name="V", instruction="watch",
                mcp_server=MagicMock(), ha_client=MagicMock(),
                entities=[], vision_enabled=True,
                ollama_model="deepseek-r1:8b",
            )

    @pytest.mark.asyncio
    async def test_vision_ollama_keep_alive_minus_one(self):
        """VisionAgent _analyze_with_ollama_text must pass keep_alive=-1."""
        agent = self._make_vision_agent()
        chat_kwargs = {}

        def _capture(*args, **kwargs):
            chat_kwargs.update(kwargs)
            return {"message": {"content": "clear"}}

        agent._ollama_client = MagicMock()
        agent._ollama_client.chat.side_effect = _capture

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value=[])
        agent._ha_provider = lambda: mock_ha

        with patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.vision_agent.asyncio.sleep"):
            await agent._analyze_with_ollama_text("camera.front")

        assert chat_kwargs.get("keep_alive") == -1, (
            f"VisionAgent keep_alive={chat_kwargs.get('keep_alive')!r}, expected -1"
        )

    @pytest.mark.asyncio
    async def test_vision_ollama_num_ctx_2048(self):
        """VisionAgent _analyze_with_ollama_text must use num_ctx=2048."""
        agent = self._make_vision_agent()
        captured_options = {}

        def _capture(model, messages, options, **kwargs):
            captured_options.update(options)
            return {"message": {"content": "clear"}}

        agent._ollama_client = MagicMock()
        agent._ollama_client.chat.side_effect = _capture

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value=[])
        agent._ha_provider = lambda: mock_ha

        with patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.vision_agent.asyncio.sleep"):
            await agent._analyze_with_ollama_text("camera.front")

        assert captured_options.get("num_ctx") == 2048, (
            f"VisionAgent num_ctx={captured_options.get('num_ctx')}, expected 2048"
        )

    @pytest.mark.asyncio
    async def test_vision_ollama_repeat_penalty_1_0(self):
        """VisionAgent _analyze_with_ollama_text must set repeat_penalty=1.0."""
        agent = self._make_vision_agent()
        captured_options = {}

        def _capture(model, messages, options, **kwargs):
            captured_options.update(options)
            return {"message": {"content": "clear"}}

        agent._ollama_client = MagicMock()
        agent._ollama_client.chat.side_effect = _capture

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value=[])
        agent._ha_provider = lambda: mock_ha

        with patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.vision_agent.asyncio.sleep"):
            await agent._analyze_with_ollama_text("camera.front")

        assert captured_options.get("repeat_penalty") == 1.0, (
            f"VisionAgent repeat_penalty={captured_options.get('repeat_penalty')}, expected 1.0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. run_decision_loop warmup
# ─────────────────────────────────────────────────────────────────────────────

class TestRunDecisionLoopWarmup:

    @pytest.mark.asyncio
    async def test_warmup_call_fires_before_main_loop(self):
        """
        run_decision_loop must call _call_llm once before the main while-loop
        (the warmup), then call gather_context / decide as normal.
        The warmup call happens BEFORE gather_context.
        """
        agent = _make_base_agent()

        mock_ha = MagicMock()
        mock_ha.connected = True
        agent._ha_provider = lambda: mock_ha

        call_order = []

        async def _fake_call_llm(prompt, **kwargs):
            call_order.append(("llm", prompt[:10]))
            return "ok"

        async def _fake_gather():
            call_order.append(("gather",))
            raise KeyboardInterrupt  # stop after first real cycle

        agent._call_llm = _fake_call_llm
        agent.gather_context = _fake_gather
        agent.log_decision = MagicMock()

        with patch("agents.base_agent.asyncio.sleep"):
            try:
                await agent.run_decision_loop()
            except KeyboardInterrupt:
                pass

        assert len(call_order) >= 2, f"Expected warmup + gather, got {call_order}"
        # First call must be the warmup (prompt starts with "ping")
        assert call_order[0][0] == "llm" and call_order[0][1].startswith("ping"), (
            f"First call was not warmup: {call_order}"
        )
        # Second call must be gather_context
        assert call_order[1][0] == "gather", (
            f"gather_context not called after warmup: {call_order}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10-12. UniversalAgent services cache
# ─────────────────────────────────────────────────────────────────────────────

class TestUniversalAgentServicesCache:

    def test_services_cache_initialises_as_none(self):
        """_services_cache must be None at construction (lazy fetch)."""
        agent = _make_universal_agent()
        assert agent._services_cache is None
        assert agent._services_cache_expires is None

    @pytest.mark.asyncio
    async def test_second_decide_does_not_refetch_services(self):
        """
        After the first decide() populates the services cache, a second decide()
        must NOT call ha_client.get_services() again.
        """
        agent = _make_universal_agent(entities=["light.hall"])

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value={"entity_id": "light.hall", "state": "on", "attributes": {}})
        mock_ha.get_services = AsyncMock(return_value={"light": {"turn_on": {}, "turn_off": {}}})
        agent._ha_provider = lambda: mock_ha

        context = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "state_description": "- light.hall: on",
            "instruction": "manage lights",
        }

        async def _fake_llm(prompt, **kwargs):
            return '{"reasoning": "ok", "actions": []}'

        agent._call_llm = _fake_llm

        # First decide — must fetch services
        await agent.decide(context)
        assert mock_ha.get_services.call_count == 1, "First decide should fetch services once"

        # Second decide — must use cache, NOT fetch again
        await agent.decide(context)
        assert mock_ha.get_services.call_count == 1, (
            f"get_services called {mock_ha.get_services.call_count} times — cache not working"
        )

    @pytest.mark.asyncio
    async def test_services_refetched_after_cache_expires(self):
        """
        Once the services cache expires, the next decide() must re-fetch from HA.
        """
        agent = _make_universal_agent(entities=["light.hall"])

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value={"entity_id": "light.hall", "state": "on", "attributes": {}})
        mock_ha.get_services = AsyncMock(return_value={"light": {"turn_on": {}, "turn_off": {}}})
        agent._ha_provider = lambda: mock_ha

        context = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "state_description": "- light.hall: on",
            "instruction": "manage lights",
        }

        async def _fake_llm(prompt, **kwargs):
            return '{"reasoning": "ok", "actions": []}'

        agent._call_llm = _fake_llm

        # First decide — populates cache
        await agent.decide(context)

        # Expire the cache manually (simulate 11 minutes passing)
        agent._services_cache_expires = datetime.now() - timedelta(seconds=1)

        # Second decide — cache expired, must re-fetch
        await agent.decide(context)
        assert mock_ha.get_services.call_count == 2, (
            f"Expected 2 service fetches (cache expired), got {mock_ha.get_services.call_count}"
        )
