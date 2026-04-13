"""
Tests for v1.2.1 — revert of event-driven + per-model semaphores.

Verifies that:
1.  Single global LLM semaphore is used (no per-model dict)
2.  _get_llm_semaphore() takes no arguments and returns the same Semaphore
3.  Concurrent calls to different "model" agents still serialise through the one semaphore
4.  VisionAgent Gemini path uses the global semaphore (no key argument)
5.  VisionAgent Ollama fallback path uses the global semaphore (no key argument)
6.  BaseAgent has no _event_driven, _trigger_event, or _on_entity_changed
7.  UniversalAgent has no event_driven parameter
8.  run_decision_loop is plain polling (no subscribe_entities call)
9.  Default model_name in BaseAgent is deepseek-r1:8b
10. Default decision_interval in BaseAgent is 300
11. agents.yaml lighting uses deepseek-r1:8b and 300s interval
12. agents.yaml security uses deepseek-r1:8b and 300s interval
13. config.json fast_model is deepseek-r1:8b
14. config.json smart_model is deepseek-r1:8b
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import yaml

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


def _reset_semaphore():
    import agents.base_agent as ba
    ba._LLM_SEMAPHORE = None


def _make_base_agent(model_name="deepseek-r1:8b", decision_interval=300):
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
            decision_interval=decision_interval,
        )
    return agent


async def _fake_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 1 & 2. Single global semaphore
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalSemaphore:

    def setup_method(self):
        _reset_semaphore()

    def teardown_method(self):
        _reset_semaphore()

    def test_get_llm_semaphore_takes_no_args(self):
        """_get_llm_semaphore() must be callable without arguments."""
        from agents.base_agent import _get_llm_semaphore
        sem = _get_llm_semaphore()
        assert isinstance(sem, asyncio.Semaphore)

    def test_singleton_returns_same_object(self):
        """Every call must return the exact same Semaphore instance."""
        from agents.base_agent import _get_llm_semaphore
        s1 = _get_llm_semaphore()
        s2 = _get_llm_semaphore()
        assert s1 is s2, "Global semaphore is not a singleton"

    def test_no_llm_semaphores_dict(self):
        """Module must NOT have a _LLM_SEMAPHORES dict (per-model approach removed)."""
        import agents.base_agent as ba
        assert not hasattr(ba, "_LLM_SEMAPHORES"), (
            "_LLM_SEMAPHORES dict still present — per-model semaphores not fully removed"
        )

    def test_llm_semaphore_is_none_before_first_call(self):
        """_LLM_SEMAPHORE starts as None and is lazily created."""
        import agents.base_agent as ba
        ba._LLM_SEMAPHORE = None
        assert ba._LLM_SEMAPHORE is None
        ba._get_llm_semaphore()
        assert ba._LLM_SEMAPHORE is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Concurrent agents serialise through the single semaphore
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialisation:

    def setup_method(self):
        _reset_semaphore()

    def teardown_method(self):
        _reset_semaphore()

    @pytest.mark.asyncio
    async def test_different_model_agents_serialise(self):
        """
        Two agents with different model names must still queue behind the one
        global semaphore — ensuring only one Ollama call runs at a time.
        """
        a1 = _make_base_agent(model_name="deepseek-r1:8b")
        a2 = _make_base_agent(model_name="mistral:7b-instruct")
        a1.ollama_client = MagicMock()
        a2.ollama_client = MagicMock()
        a1.ollama_client.chat.return_value = {"message": {"content": "ok"}}
        a2.ollama_client.chat.return_value = {"message": {"content": "ok"}}

        timeline = []

        async def _fake_slow_thread(func, *args, **kwargs):
            timeline.append("start")
            await asyncio.sleep(0.05)
            timeline.append("end")
            return {"message": {"content": "ok"}}

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_slow_thread):
            await asyncio.gather(a1._call_llm("p"), a2._call_llm("p"))

        assert timeline == ["start", "end", "start", "end"], (
            f"LLM calls from different-model agents overlapped: {timeline}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. VisionAgent uses global semaphore (no key argument)
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionAgentGlobalSemaphore:

    def setup_method(self):
        _reset_semaphore()

    def teardown_method(self):
        _reset_semaphore()

    def _make_vision_agent(self, ollama_model="deepseek-r1:8b"):
        with patch("agents.vision_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
            from agents.vision_agent import VisionAgent
            return VisionAgent(
                agent_id="v", name="V", instruction="watch",
                mcp_server=MagicMock(), ha_client=MagicMock(),
                entities=[], gemini_api_key="", vision_enabled=False,
                ollama_model=ollama_model,
            )

    @pytest.mark.asyncio
    async def test_gemini_path_acquires_global_semaphore(self):
        """_analyze_with_gemini must acquire _get_llm_semaphore() with no args."""
        agent = self._make_vision_agent()

        # Track calls to _get_llm_semaphore — must be called with no args
        semaphore_call_args = []

        original_get_sem = None

        import agents.vision_agent as va
        original_get_sem_ref = [None]

        def mock_get_sem(*args, **kwargs):
            semaphore_call_args.append((args, kwargs))
            return asyncio.Semaphore(1)

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="clear")
        agent._vision_model = mock_model

        mock_ha = MagicMock()
        mock_ha.get_camera_snapshot = AsyncMock(return_value=b"\xff\xd8\xff")
        agent._ha_provider = lambda: mock_ha

        import io
        mock_pil = MagicMock()
        with patch.object(va, "PIL", mock_pil, create=True), \
             patch.object(va, "_io", io, create=True), \
             patch("agents.vision_agent._get_llm_semaphore", side_effect=mock_get_sem), \
             patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread):
            await agent._analyze_with_gemini("camera.front")

        assert semaphore_call_args, "_get_llm_semaphore not called from Gemini path"
        args, kwargs = semaphore_call_args[0]
        assert args == () and kwargs == {}, (
            f"_get_llm_semaphore called with args={args}, kwargs={kwargs} — "
            "should be called with no arguments (global semaphore)"
        )

    @pytest.mark.asyncio
    async def test_ollama_fallback_acquires_global_semaphore(self):
        """_analyze_with_ollama_text must acquire _get_llm_semaphore() with no args."""
        agent = self._make_vision_agent(ollama_model="deepseek-r1:8b")

        semaphore_call_args = []

        def mock_get_sem(*args, **kwargs):
            semaphore_call_args.append((args, kwargs))
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

        assert semaphore_call_args, "_get_llm_semaphore not called from Ollama path"
        args, kwargs = semaphore_call_args[0]
        assert args == () and kwargs == {}, (
            f"_get_llm_semaphore called with args={args}, kwargs={kwargs} — "
            "should be called with no arguments (global semaphore)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. No event-driven attributes on BaseAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestNoEventDrivenOnBaseAgent:

    def test_no_event_driven_attribute(self):
        """BaseAgent must NOT have _event_driven after the revert."""
        agent = _make_base_agent()
        assert not hasattr(agent, "_event_driven"), (
            "_event_driven still present — event-driven code not fully removed from BaseAgent"
        )

    def test_no_trigger_event_attribute(self):
        """BaseAgent must NOT have _trigger_event."""
        agent = _make_base_agent()
        assert not hasattr(agent, "_trigger_event"), (
            "_trigger_event still present — event-driven code not fully removed"
        )

    def test_no_on_entity_changed_method(self):
        """BaseAgent must NOT have _on_entity_changed."""
        agent = _make_base_agent()
        assert not hasattr(agent, "_on_entity_changed"), (
            "_on_entity_changed still present — event-driven code not fully removed"
        )

    def test_no_last_decision_at_attribute(self):
        """BaseAgent must NOT have _last_decision_at."""
        agent = _make_base_agent()
        assert not hasattr(agent, "_last_decision_at"), (
            "_last_decision_at still present — event-driven code not fully removed"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. UniversalAgent has no event_driven parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestUniversalAgentNoEventDriven:

    def _make_ua(self, **kwargs):
        from agents.universal_agent import UniversalAgent
        mcp = MagicMock()
        mcp.dry_run = False
        with patch("agents.base_agent.ollama.Client"), patch("pathlib.Path.mkdir"):
            return UniversalAgent(
                agent_id="ua",
                name="UA",
                instruction="Test",
                mcp_server=mcp,
                ha_client=MagicMock(),
                entities=[],
                **kwargs,
            )

    def test_universal_agent_accepts_no_event_driven_kwarg(self):
        """UniversalAgent must be constructable without event_driven."""
        agent = self._make_ua()
        assert agent is not None

    def test_universal_agent_rejects_event_driven_kwarg(self):
        """event_driven kwarg must raise TypeError — parameter removed."""
        with pytest.raises(TypeError):
            self._make_ua(event_driven=True)

    def test_no_event_driven_on_universal_agent(self):
        """UniversalAgent instance must not have _event_driven attribute."""
        agent = self._make_ua()
        assert not hasattr(agent, "_event_driven")


# ─────────────────────────────────────────────────────────────────────────────
# 8. run_decision_loop is plain polling (no subscribe_entities)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlainPollingLoop:

    @pytest.mark.asyncio
    async def test_run_decision_loop_does_not_call_subscribe_entities(self):
        """run_decision_loop must NOT call ha_client.subscribe_entities."""
        agent = _make_base_agent()

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.subscribe_entities = AsyncMock()
        agent._ha_provider = lambda: mock_ha

        agent.gather_context = AsyncMock(side_effect=KeyboardInterrupt)

        with patch("agents.base_agent.asyncio.sleep"):
            try:
                await agent.run_decision_loop()
            except KeyboardInterrupt:
                pass

        mock_ha.subscribe_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_decision_loop_sleeps_decision_interval(self):
        """After each decision, run_decision_loop must sleep for decision_interval seconds."""
        agent = _make_base_agent(decision_interval=300)

        mock_ha = MagicMock()
        mock_ha.connected = True
        agent._ha_provider = lambda: mock_ha

        sleep_calls = []

        async def _fake_sleep(seconds):
            sleep_calls.append(seconds)
            # Stop after we've seen a 300s sleep (the decision interval sleep)
            if seconds == 300:
                raise KeyboardInterrupt

        # Patch _call_llm so the warmup succeeds immediately (no retry sleeps)
        agent._call_llm = AsyncMock(return_value="ok")
        agent.gather_context = AsyncMock(return_value={})
        agent.decide = AsyncMock(return_value={"reasoning": "ok", "actions": []})
        agent.execute = AsyncMock(return_value=[])
        agent.log_decision = MagicMock()

        with patch("agents.base_agent.asyncio.sleep", side_effect=_fake_sleep):
            try:
                await agent.run_decision_loop()
            except KeyboardInterrupt:
                pass

        decision_sleeps = [s for s in sleep_calls if s == 300]
        assert decision_sleeps, (
            f"run_decision_loop never slept for decision_interval=300. Sleeps: {sleep_calls}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9 & 10. Default model and interval
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaults:

    def test_default_model_is_deepseek(self):
        """BaseAgent default model_name must be deepseek-r1:8b."""
        import inspect
        from agents.base_agent import BaseAgent
        sig = inspect.signature(BaseAgent.__init__)
        default = sig.parameters["model_name"].default
        assert default == "deepseek-r1:8b", (
            f"Default model_name is '{default}', expected 'deepseek-r1:8b'"
        )

    def test_default_interval_is_300(self):
        """BaseAgent default decision_interval must be 300."""
        import inspect
        from agents.base_agent import BaseAgent
        sig = inspect.signature(BaseAgent.__init__)
        default = sig.parameters["decision_interval"].default
        assert default == 300, (
            f"Default decision_interval is {default}, expected 300"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 11 & 12. agents.yaml sanity checks
# ─────────────────────────────────────────────────────────────────────────────

_AGENTS_YAML = Path(__file__).parent.parent.parent / "agents.yaml"


class TestAgentsYaml:

    @pytest.fixture(autouse=True)
    def _load(self):
        if not _AGENTS_YAML.exists():
            pytest.skip("agents.yaml not found")
        with open(_AGENTS_YAML) as f:
            self.cfg = yaml.safe_load(f)
        self.agents = {a["id"]: a for a in self.cfg.get("agents", [])}

    def test_lighting_uses_deepseek(self):
        assert "lighting" in self.agents
        assert self.agents["lighting"]["model"] == "deepseek-r1:8b", (
            f"lighting model is {self.agents['lighting']['model']!r}, expected 'deepseek-r1:8b'"
        )

    def test_lighting_no_event_driven(self):
        assert "event_driven" not in self.agents.get("lighting", {}), (
            "lighting still has event_driven key"
        )

    def test_lighting_interval_300(self):
        assert self.agents["lighting"].get("decision_interval") == 300, (
            f"lighting interval is {self.agents['lighting'].get('decision_interval')}, expected 300"
        )

    def test_security_uses_deepseek(self):
        assert "security" in self.agents
        assert self.agents["security"]["model"] == "deepseek-r1:8b", (
            f"security model is {self.agents['security']['model']!r}, expected 'deepseek-r1:8b'"
        )

    def test_security_no_event_driven(self):
        assert "event_driven" not in self.agents.get("security", {}), (
            "security still has event_driven key"
        )

    def test_security_interval_300(self):
        assert self.agents["security"].get("decision_interval") == 300, (
            f"security interval is {self.agents['security'].get('decision_interval')}, expected 300"
        )

    def test_no_agent_uses_fast_alias(self):
        for agent in self.cfg.get("agents", []):
            assert agent.get("model") != "fast", (
                f"Agent {agent['id']} still uses 'fast' model alias"
            )

    def test_no_agent_uses_smart_alias(self):
        for agent in self.cfg.get("agents", []):
            assert agent.get("model") != "smart", (
                f"Agent {agent['id']} still uses 'smart' model alias"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 13 & 14. config.json model defaults
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_JSON = Path(__file__).parent.parent.parent / "config.json"


class TestConfigJson:

    @pytest.fixture(autouse=True)
    def _load(self):
        if not _CONFIG_JSON.exists():
            pytest.skip("config.json not found")
        with open(_CONFIG_JSON) as f:
            self.cfg = json.load(f)

    def test_fast_model_is_deepseek(self):
        assert self.cfg["options"]["fast_model"] == "deepseek-r1:8b", (
            f"fast_model is {self.cfg['options']['fast_model']!r}, expected 'deepseek-r1:8b'"
        )

    def test_smart_model_is_deepseek(self):
        assert self.cfg["options"]["smart_model"] == "deepseek-r1:8b", (
            f"smart_model is {self.cfg['options']['smart_model']!r}, expected 'deepseek-r1:8b'"
        )

    def test_orchestrator_model_is_deepseek(self):
        assert self.cfg["options"]["orchestrator_model"] == "deepseek-r1:8b", (
            f"orchestrator_model is {self.cfg['options']['orchestrator_model']!r}, "
            "expected 'deepseek-r1:8b'"
        )

    def test_version_is_at_least_1_2_1(self):
        from packaging.version import Version
        assert Version(self.cfg["version"]) >= Version("1.2.1"), (
            f"config.json version {self.cfg['version']!r} is older than 1.2.1"
        )
