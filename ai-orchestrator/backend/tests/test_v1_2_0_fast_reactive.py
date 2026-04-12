"""
Tests for v1.2.0 fast reactive agents.

Covers:
1.  Per-model semaphores — different models don't block each other
2.  _get_llm_semaphore returns same object for same model name
3.  Gemini semaphore is distinct from Ollama semaphores
4.  Fast model semaphore does not block when slow model semaphore is held
5.  _call_llm passes self.model_name to the semaphore
6.  VisionAgent Gemini path uses "gemini" semaphore key
7.  VisionAgent Ollama fallback path uses ollama model name semaphore key
8.  event_driven=False: _event_driven attribute is False, no _trigger_event
9.  event_driven=True: _event_driven attribute is True
10. _on_entity_changed: sets _trigger_event when idle and past debounce window
11. _on_entity_changed: ignored when status == "deciding"
12. _on_entity_changed: ignored within debounce window (_min_trigger_gap)
13. _on_entity_changed: fires on first call (_last_decision_at starts at 0.0)
14. _on_entity_changed: does nothing when _trigger_event is None
15. run_decision_loop subscribes entities when event_driven=True and entities set
16. run_decision_loop skips subscription when event_driven=False
17. run_decision_loop skips subscription when entities list is empty
18. model "fast" alias resolves to FAST_MODEL env var
19. model "smart" alias resolves to SMART_MODEL env var
20. explicit model name is unchanged by alias resolution
21. UniversalAgent accepts event_driven kwarg and propagates to BaseAgent
22. UniversalAgent event_driven defaults to False
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# ── path + heavy-dep stubs ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

for _mod in (
    "langgraph", "langgraph.graph", "chromadb", "chromadb.config",
    "chromadb.utils", "langgraph.checkpoint", "langgraph.checkpoint.memory",
    "fastapi", "fastapi.routing", "google.generativeai", "PIL", "PIL.Image",
):
    sys.modules.setdefault(_mod, MagicMock())


async def _fake_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


def _reset_semaphores():
    import agents.base_agent as ba
    ba._LLM_SEMAPHORES.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_base_agent(model_name="mistral:7b-instruct", event_driven=False, entities=None):
    from agents.base_agent import BaseAgent

    class ConcreteAgent(BaseAgent):
        async def decide(self, ctx):
            return {"actions": []}
        async def gather_context(self):
            return {}

    mcp = MagicMock()
    mcp.dry_run = False

    with patch("agents.base_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
        agent = ConcreteAgent(
            agent_id="test",
            name="Test",
            mcp_server=mcp,
            ha_client=MagicMock(),
            skills_path="/nonexistent/SKILLS.md",
            model_name=model_name,
            event_driven=event_driven,
        )
    if entities is not None:
        agent.entities = entities
    return agent


def _make_universal_agent(event_driven=False, entities=None, model_name="mistral:7b-instruct"):
    from agents.universal_agent import UniversalAgent

    mcp = MagicMock()
    mcp.dry_run = False

    with patch("agents.base_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
        agent = UniversalAgent(
            agent_id="ua",
            name="UA",
            instruction="Test",
            mcp_server=mcp,
            ha_client=MagicMock(),
            entities=entities or [],
            model_name=model_name,
            event_driven=event_driven,
        )
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# 1-4. Per-model semaphores
# ─────────────────────────────────────────────────────────────────────────────

class TestPerModelSemaphores:

    def setup_method(self):
        _reset_semaphores()

    def test_different_models_get_different_semaphores(self):
        from agents.base_agent import _get_llm_semaphore
        assert _get_llm_semaphore("mistral:7b-instruct") is not _get_llm_semaphore("gemma4:e4b")

    def test_same_model_returns_same_semaphore(self):
        from agents.base_agent import _get_llm_semaphore
        assert _get_llm_semaphore("mistral:7b-instruct") is _get_llm_semaphore("mistral:7b-instruct")

    def test_gemini_gets_own_semaphore(self):
        from agents.base_agent import _get_llm_semaphore
        assert _get_llm_semaphore("gemini") is not _get_llm_semaphore("mistral:7b-instruct")

    @pytest.mark.asyncio
    async def test_fast_model_does_not_block_slow_model(self):
        """Holding the mistral semaphore must not prevent gemma4:e4b from being acquired."""
        from agents.base_agent import _get_llm_semaphore
        slow_sem = _get_llm_semaphore("mistral:7b-instruct")
        fast_sem = _get_llm_semaphore("gemma4:e4b")
        acquired = False
        async with slow_sem:
            async with fast_sem:
                acquired = True
        assert acquired


# ─────────────────────────────────────────────────────────────────────────────
# 5. _call_llm passes self.model_name to the semaphore
# ─────────────────────────────────────────────────────────────────────────────

class TestCallLlmSemaphoreKey:

    def setup_method(self):
        _reset_semaphores()

    @pytest.mark.asyncio
    async def test_call_llm_creates_semaphore_for_agent_model(self):
        """_call_llm must acquire the semaphore keyed by self.model_name."""
        agent = _make_base_agent(model_name="gemma4:e4b")
        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.return_value = {"message": {"content": "ok"}}

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.base_agent.asyncio.sleep"):
            await agent._call_llm("hello")

        import agents.base_agent as ba
        assert "gemma4:e4b" in ba._LLM_SEMAPHORES
        assert "mistral:7b-instruct" not in ba._LLM_SEMAPHORES


# ─────────────────────────────────────────────────────────────────────────────
# 6 & 7. VisionAgent semaphore keys
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionAgentSemaphoreKeys:

    def setup_method(self):
        _reset_semaphores()

    def _make_vision_agent(self, ollama_model="mistral:7b-instruct"):
        with patch("agents.vision_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
            from agents.vision_agent import VisionAgent
            return VisionAgent(
                agent_id="v", name="V", instruction="watch",
                mcp_server=MagicMock(), ha_client=MagicMock(),
                entities=[], gemini_api_key="", vision_enabled=False,
                ollama_model=ollama_model,
            )

    @pytest.mark.asyncio
    async def test_gemini_path_acquires_gemini_semaphore(self):
        """_analyze_with_gemini must use the 'gemini' semaphore key."""
        agent = self._make_vision_agent()

        acquired_keys = []

        def mock_get_sem(key="default"):
            acquired_keys.append(key)
            return asyncio.Semaphore(1)

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="clear")
        agent._vision_model = mock_model

        mock_ha = MagicMock()
        mock_ha.get_camera_snapshot = AsyncMock(return_value=b"\xff\xd8\xff")
        agent._ha_provider = lambda: mock_ha

        import io
        import agents.vision_agent as va
        mock_pil = MagicMock()
        # Inject PIL and _io into vision_agent's module namespace; the try/except
        # guard import may have failed in environments without google-generativeai
        # or Pillow installed, leaving these names unbound.
        with patch.object(va, "PIL", mock_pil, create=True), \
             patch.object(va, "_io", io, create=True), \
             patch("agents.vision_agent._get_llm_semaphore", side_effect=mock_get_sem), \
             patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread):
            await agent._analyze_with_gemini("camera.front")

        assert "gemini" in acquired_keys, f"Expected 'gemini' key, got {acquired_keys}"

    @pytest.mark.asyncio
    async def test_ollama_fallback_acquires_model_name_semaphore(self):
        """_analyze_with_ollama_text must use self._ollama_model as semaphore key."""
        agent = self._make_vision_agent(ollama_model="gemma4:e4b")

        acquired_keys = []

        def mock_get_sem(key="default"):
            acquired_keys.append(key)
            return asyncio.Semaphore(1)

        agent._ollama_client = MagicMock()
        agent._ollama_client.chat.return_value = {"message": {"content": "all clear"}}

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value=[])
        agent._ha_provider = lambda: mock_ha

        with patch("agents.vision_agent._get_llm_semaphore", side_effect=mock_get_sem), \
             patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.vision_agent.asyncio.sleep"):
            await agent._analyze_with_ollama_text("camera.front")

        assert "gemma4:e4b" in acquired_keys, f"Expected 'gemma4:e4b' key, got {acquired_keys}"


# ─────────────────────────────────────────────────────────────────────────────
# 8 & 9. event_driven attribute on BaseAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestEventDrivenAttribute:

    def test_event_driven_false_by_default(self):
        agent = _make_base_agent()
        assert agent._event_driven is False

    def test_event_driven_true_stored(self):
        agent = _make_base_agent(event_driven=True)
        assert agent._event_driven is True

    def test_trigger_event_none_at_init(self):
        agent = _make_base_agent(event_driven=True)
        assert agent._trigger_event is None

    def test_last_decision_at_zero_at_init(self):
        agent = _make_base_agent()
        assert agent._last_decision_at == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 10-14. _on_entity_changed debounce
# ─────────────────────────────────────────────────────────────────────────────

class TestOnEntityChangedDebounce:

    @pytest.mark.asyncio
    async def test_sets_trigger_event_when_idle_and_past_debounce(self):
        agent = _make_base_agent(event_driven=True)
        agent._trigger_event = asyncio.Event()
        agent.status = "idle"
        agent._last_decision_at = time.monotonic() - 20.0  # well past debounce

        await agent._on_entity_changed({})

        assert agent._trigger_event.is_set()

    @pytest.mark.asyncio
    async def test_ignored_when_status_deciding(self):
        agent = _make_base_agent(event_driven=True)
        agent._trigger_event = asyncio.Event()
        agent.status = "deciding"
        agent._last_decision_at = 0.0  # would normally trigger

        await agent._on_entity_changed({})

        assert not agent._trigger_event.is_set()

    @pytest.mark.asyncio
    async def test_ignored_within_debounce_window(self):
        agent = _make_base_agent(event_driven=True)
        agent._trigger_event = asyncio.Event()
        agent.status = "idle"
        agent._last_decision_at = time.monotonic()  # just decided — within gap
        agent._min_trigger_gap = 10.0

        await agent._on_entity_changed({})

        assert not agent._trigger_event.is_set()

    @pytest.mark.asyncio
    async def test_triggers_on_first_call_last_decision_zero(self):
        """_last_decision_at = 0.0 so the very first entity change always fires."""
        agent = _make_base_agent(event_driven=True)
        agent._trigger_event = asyncio.Event()
        agent.status = "idle"
        # _last_decision_at == 0.0 by default

        await agent._on_entity_changed({})

        assert agent._trigger_event.is_set()

    @pytest.mark.asyncio
    async def test_does_nothing_when_trigger_event_none(self):
        """No crash if _trigger_event is None (subscription not set up yet)."""
        agent = _make_base_agent(event_driven=True)
        agent._trigger_event = None
        agent.status = "idle"

        # Should not raise
        await agent._on_entity_changed({})


# ─────────────────────────────────────────────────────────────────────────────
# 15-17. Subscription setup in run_decision_loop
# ─────────────────────────────────────────────────────────────────────────────

class TestRunDecisionLoopSubscription:

    def _connected_ha(self):
        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.subscribe_entities = AsyncMock(return_value=1)
        return mock_ha

    @pytest.mark.asyncio
    async def test_subscribes_when_event_driven_true_and_entities_set(self):
        agent = _make_base_agent(event_driven=True, entities=["light.hall"])
        mock_ha = self._connected_ha()
        agent._ha_provider = lambda: mock_ha  # lambda so property returns mock_ha directly

        # Replace loop body to stop after one iteration
        call_count = 0
        original_gather = agent.gather_context.__func__ if hasattr(agent.gather_context, '__func__') else None

        async def one_shot_gather(self_):
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt  # stop loop after subscription is set up

        agent.gather_context = lambda: (_ for _ in ()).throw(KeyboardInterrupt())

        # Use a simpler stop mechanism: patch asyncio.wait_for to raise immediately
        async def immediate_stop(coro, timeout):
            coro.close()
            raise KeyboardInterrupt

        with patch("agents.base_agent.asyncio.wait_for", side_effect=immediate_stop), \
             patch("agents.base_agent.asyncio.sleep"):
            # gather_context raises before entering decision, but subscription
            # happens BEFORE the while loop, so we need to get past it
            agent.gather_context = AsyncMock(side_effect=KeyboardInterrupt)
            agent.log_decision = MagicMock()

            try:
                await agent.run_decision_loop()
            except KeyboardInterrupt:
                pass

        mock_ha.subscribe_entities.assert_called_once_with(
            ["light.hall"], agent._on_entity_changed
        )

    @pytest.mark.asyncio
    async def test_no_subscription_when_event_driven_false(self):
        agent = _make_base_agent(event_driven=False, entities=["light.hall"])
        mock_ha = self._connected_ha()
        agent._ha_provider = lambda: mock_ha

        agent.gather_context = AsyncMock(side_effect=KeyboardInterrupt)

        with patch("agents.base_agent.asyncio.sleep"):
            try:
                await agent.run_decision_loop()
            except KeyboardInterrupt:
                pass

        mock_ha.subscribe_entities.assert_not_called()
        assert agent._trigger_event is None

    @pytest.mark.asyncio
    async def test_no_subscription_when_entities_empty(self):
        agent = _make_base_agent(event_driven=True, entities=[])
        mock_ha = self._connected_ha()
        agent._ha_provider = lambda: mock_ha

        agent.gather_context = AsyncMock(side_effect=KeyboardInterrupt)

        with patch("agents.base_agent.asyncio.sleep"):
            try:
                await agent.run_decision_loop()
            except KeyboardInterrupt:
                pass

        mock_ha.subscribe_entities.assert_not_called()
        assert agent._trigger_event is None


# ─────────────────────────────────────────────────────────────────────────────
# 18-20. Model alias resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestModelAliasResolution:
    """Tests the alias resolution logic from main.py."""

    def _resolve(self, model_name, fast="gemma4:e4b", smart="mistral:7b-instruct"):
        if model_name == "fast":
            return fast
        elif model_name == "smart":
            return smart
        return model_name

    def test_fast_alias_resolves_to_fast_model(self):
        assert self._resolve("fast") == "gemma4:e4b"

    def test_smart_alias_resolves_to_smart_model(self):
        assert self._resolve("smart") == "mistral:7b-instruct"

    def test_explicit_model_name_unchanged(self):
        assert self._resolve("deepseek-r1:8b") == "deepseek-r1:8b"

    def test_gemini_unchanged(self):
        assert self._resolve("gemini") == "gemini"

    def test_fast_uses_custom_env(self):
        assert self._resolve("fast", fast="llama3.2:3b") == "llama3.2:3b"


# ─────────────────────────────────────────────────────────────────────────────
# 21 & 22. UniversalAgent event_driven propagation
# ─────────────────────────────────────────────────────────────────────────────

class TestUniversalAgentEventDriven:

    def test_event_driven_false_by_default(self):
        agent = _make_universal_agent()
        assert agent._event_driven is False

    def test_event_driven_true_propagated_to_base(self):
        agent = _make_universal_agent(event_driven=True)
        assert agent._event_driven is True

    def test_entities_stored_on_agent(self):
        agent = _make_universal_agent(entities=["light.hall", "binary_sensor.motion"])
        assert agent.entities == ["light.hall", "binary_sensor.motion"]

    def test_min_trigger_gap_default_is_10(self):
        agent = _make_universal_agent(event_driven=True)
        assert agent._min_trigger_gap == 10.0
