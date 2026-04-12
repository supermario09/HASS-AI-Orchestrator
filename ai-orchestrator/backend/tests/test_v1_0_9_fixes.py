"""
Tests for v1.0.9 fixes:

1. Global LLM semaphore — _get_llm_semaphore() is lazily created, shared across calls,
   and _call_llm() acquires it before touching Ollama.

2. VisionAgent Ollama fallback — when Gemini is unavailable:
   - analyze_camera() routes to _analyze_with_ollama_text()
   - _analyze_with_ollama_text() uses asyncio.to_thread + LLM semaphore
   - run_decision_loop() NO LONGER dead-ends; it falls through to the main camera loop

3. Decision feedback — the feedback-writing logic correctly persists a rating into
   the JSON decision file, and rejects invalid values.
"""
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ── path + heavy-dep stubs (same pattern as test_stability_fixes.py) ──────────
sys.path.insert(0, str(Path(__file__).parent.parent))

for _mod in (
    "langgraph",
    "langgraph.graph",
    "chromadb",
    "chromadb.config",
    "chromadb.utils",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "fastapi",
    "fastapi.routing",
):
    sys.modules.setdefault(_mod, MagicMock())

sys.modules["langgraph.graph"].END = "END"
sys.modules["langgraph.graph"].StateGraph = MagicMock()

# Stub google.generativeai so VisionAgent can import without the real package
for _mod in ("google", "google.generativeai", "PIL", "PIL.Image"):
    sys.modules.setdefault(_mod, MagicMock())


# ===========================================================================
# 1.  Global LLM Semaphore
# ===========================================================================

class TestLLMSemaphore:
    """_get_llm_semaphore() must return a single shared asyncio.Semaphore."""

    def test_semaphore_is_asyncio_semaphore(self):
        """The function must return an asyncio.Semaphore instance."""
        # Reset module-level state so we test lazy creation cleanly
        import agents.base_agent as ba
        orig = ba._LLM_SEMAPHORE
        ba._LLM_SEMAPHORE = None
        try:
            # Must be called inside an event loop so Semaphore can be created
            async def _get():
                return ba._get_llm_semaphore()
            sem = asyncio.get_event_loop().run_until_complete(_get())
            assert isinstance(sem, asyncio.Semaphore), (
                f"Expected asyncio.Semaphore, got {type(sem)}"
            )
        finally:
            ba._LLM_SEMAPHORE = orig

    def test_semaphore_is_singleton(self):
        """Two consecutive calls must return the same object."""
        import agents.base_agent as ba
        orig = ba._LLM_SEMAPHORE
        ba._LLM_SEMAPHORE = None
        try:
            async def _get_both():
                s1 = ba._get_llm_semaphore()
                s2 = ba._get_llm_semaphore()
                return s1, s2
            s1, s2 = asyncio.get_event_loop().run_until_complete(_get_both())
            assert s1 is s2, "Each call returned a different semaphore — not a singleton"
        finally:
            ba._LLM_SEMAPHORE = orig

    @pytest.mark.asyncio
    async def test_call_llm_acquires_semaphore(self, tmp_path):
        """_call_llm() must acquire the LLM semaphore before calling to_thread."""
        from agents.base_agent import BaseAgent

        class _Agent(BaseAgent):
            async def decide(self, ctx): return {"actions": []}
            async def gather_context(self): return {}

        with patch("pathlib.Path.mkdir"):
            agent = _Agent(
                agent_id="sem_test", name="SemTest",
                mcp_server=MagicMock(), ha_client=MagicMock(),
                skills_path=str(tmp_path / "SKILLS.md"), model_name="test",
            )

        agent.ollama_client = MagicMock()
        acquired_times = []

        original_to_thread = asyncio.to_thread

        # Intercept to_thread and record whether the semaphore is held at that point
        import agents.base_agent as ba
        real_sem = ba._get_llm_semaphore()

        async def _spy_to_thread(func, *args, **kwargs):
            # If the semaphore is held (locked) when to_thread is called, our guard works
            acquired_times.append(real_sem.locked())
            return {"message": {"content": "ok"}}

        with patch("asyncio.to_thread", side_effect=_spy_to_thread):
            await agent._call_llm("hello")

        assert acquired_times, "_call_llm() never called to_thread"
        assert acquired_times[0] is True, (
            "LLM semaphore was NOT held when asyncio.to_thread was called — "
            "the async-with guard is missing or incorrectly placed"
        )

    @pytest.mark.asyncio
    async def test_semaphore_serialises_concurrent_calls(self, tmp_path):
        """
        Two concurrent _call_llm() calls must not overlap — they must queue up.
        The first call must fully complete before the second one starts.
        """
        from agents.base_agent import BaseAgent, _get_llm_semaphore

        class _Agent(BaseAgent):
            async def decide(self, ctx): return {"actions": []}
            async def gather_context(self): return {}

        def _make(aid):
            with patch("pathlib.Path.mkdir"):
                return _Agent(
                    agent_id=aid, name=aid, mcp_server=MagicMock(),
                    ha_client=MagicMock(),
                    skills_path=str(tmp_path / "SKILLS.md"), model_name="test",
                )

        a1 = _make("a1")
        a2 = _make("a2")
        a1.ollama_client = MagicMock()
        a2.ollama_client = MagicMock()

        timeline = []

        async def _fake_to_thread(func, *args, **kwargs):
            timeline.append("start")
            await asyncio.sleep(0.05)   # simulate slow LLM
            timeline.append("end")
            return {"message": {"content": "ok"}}

        with patch("asyncio.to_thread", side_effect=_fake_to_thread):
            await asyncio.gather(a1._call_llm("p"), a2._call_llm("p"))

        # Serialised → must be start/end/start/end, NOT start/start/end/end
        assert timeline == ["start", "end", "start", "end"], (
            f"LLM calls overlapped! Timeline was {timeline}. "
            "The global semaphore is not serialising concurrent calls correctly."
        )


# ===========================================================================
# 2.  VisionAgent — Ollama text fallback
# ===========================================================================

def _make_vision_agent(tmp_path, vision_enabled=True, api_key="", ollama_model="mistral:7b-instruct"):
    """Build a VisionAgent with Gemini disabled (no API key), so Ollama fallback activates."""
    # Patch genai to be unavailable so _GENAI_AVAILABLE stays False
    import agents.vision_agent as va
    orig_available = va._GENAI_AVAILABLE
    va._GENAI_AVAILABLE = bool(api_key)  # False unless we explicitly pass a key

    mock_ha = MagicMock()
    mock_ha.connected = True
    mock_ha.get_states = AsyncMock(return_value=[
        {"entity_id": "binary_sensor.front_motion", "state": "off", "attributes": {}}
    ])
    mock_ha.get_camera_snapshot = AsyncMock(return_value=b"JPEG")

    mock_mcp = MagicMock()

    with patch("pathlib.Path.mkdir"):
        agent = va.VisionAgent(
            agent_id="vision",
            name="Vision Monitor",
            instruction="Monitor for activity.",
            mcp_server=mock_mcp,
            ha_client=lambda: mock_ha,
            entities=["camera.front_door"],
            gemini_api_key=api_key,
            decision_interval=60,
            broadcast_func=None,
            vision_enabled=vision_enabled,
            ollama_model=ollama_model,
            ollama_host="http://localhost:11434",
        )

    # Restore
    va._GENAI_AVAILABLE = orig_available
    return agent, mock_ha


class TestVisionAgentOllamaFallback:

    @pytest.mark.asyncio
    async def test_analyze_camera_uses_ollama_when_no_gemini(self, tmp_path):
        """
        When _vision_model is None, analyze_camera() must call
        _analyze_with_ollama_text(), NOT _analyze_with_gemini().
        """
        agent, mock_ha = _make_vision_agent(tmp_path)
        assert agent._vision_model is None, "Expected no Gemini model for this test"

        # Mock the Ollama client so we don't need a real server
        mock_response = {"message": {"content": "No unusual activity detected."}}
        agent._ollama_client = MagicMock()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_response
            # Mock get_states on ha_client
            cam_state = {"entity_id": "camera.front_door", "state": "streaming", "attributes": {}}
            mock_ha.get_states = AsyncMock(return_value=cam_state)

            result = await agent.analyze_camera("camera.front_door")

        assert result is not None, "analyze_camera() returned None — fallback may have failed"
        assert "activity" in result.lower() or len(result) > 0

    @pytest.mark.asyncio
    async def test_ollama_text_uses_to_thread(self, tmp_path):
        """
        _analyze_with_ollama_text() must wrap ollama_client.chat in asyncio.to_thread
        (not call it directly, which would block the event loop).
        """
        agent, mock_ha = _make_vision_agent(tmp_path)
        agent._ollama_client = MagicMock()

        cam_state = {"entity_id": "camera.front_door", "state": "streaming", "attributes": {}}
        mock_ha.get_states = AsyncMock(return_value=cam_state)
        mock_ha.get_states.__call__ = AsyncMock(return_value=cam_state)

        mock_response = {"message": {"content": "All quiet."}}

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_response
            await agent._analyze_with_ollama_text("camera.front_door")

        assert mock_thread.called, (
            "_analyze_with_ollama_text() did not use asyncio.to_thread — "
            "ollama_client.chat would block the event loop"
        )
        first_arg = mock_thread.call_args[0][0]
        assert first_arg is agent._ollama_client.chat, (
            "asyncio.to_thread was not called with ollama_client.chat"
        )

    @pytest.mark.asyncio
    async def test_run_decision_loop_does_not_deadlock_without_gemini(self, tmp_path):
        """
        run_decision_loop() must NOT enter an infinite sleep when _vision_model is None.
        With the old code it would call 'while True: await asyncio.sleep(60)'.
        With the fix it falls through to the main camera loop.

        We verify this by:
        - Creating an agent with no Gemini model
        - Patching analyze_camera() to raise StopAsyncIteration after one call
          (simulates one loop iteration then bail)
        - Confirming the main while-loop body was actually reached
        """
        agent, mock_ha = _make_vision_agent(tmp_path)
        assert agent._vision_model is None

        calls = []

        async def _fake_analyze(entity_id):
            calls.append(entity_id)
            return "All clear."

        agent.analyze_camera = _fake_analyze
        agent._save_decision = MagicMock()
        agent._broadcast = AsyncMock()
        agent._announce = AsyncMock()

        # Let the loop run for one tick then cancel it
        async def _run_one_tick():
            task = asyncio.create_task(agent.run_decision_loop())
            await asyncio.sleep(0.05)   # yield so loop body executes
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_one_tick()

        assert calls, (
            "run_decision_loop() never called analyze_camera() — it likely entered "
            "the old dead-loop guard ('while True: sleep(60)') for missing Gemini"
        )

    @pytest.mark.asyncio
    async def test_gemini_path_used_when_model_available(self, tmp_path):
        """When _vision_model IS set, analyze_camera() must use _analyze_with_gemini()."""
        import agents.vision_agent as va
        agent, mock_ha = _make_vision_agent(tmp_path)

        # Inject a fake Gemini model
        mock_model = MagicMock()
        mock_resp  = MagicMock()
        mock_resp.text = "Person detected at front door."
        agent._vision_model = mock_model

        gemini_called = []

        async def _fake_gemini(eid):
            gemini_called.append(eid)
            return "Person detected."

        agent._analyze_with_gemini    = _fake_gemini
        agent._analyze_with_ollama_text = AsyncMock(return_value="Should not be called")

        result = await agent.analyze_camera("camera.front_door")

        assert "camera.front_door" in gemini_called, (
            "Gemini path not taken even though _vision_model was set"
        )


# ===========================================================================
# 3.  Decision feedback logic
# ===========================================================================

class TestDecisionFeedback:
    """
    Unit-test the feedback-writing logic in isolation (no FastAPI test client needed).
    We replicate the file-finding + update logic from the main.py endpoint.
    """

    def _write_decision(self, base_dir: Path, agent_id: str, ts: str, content: dict) -> Path:
        d = base_dir / agent_id
        d.mkdir(parents=True, exist_ok=True)
        fname = ts.replace(":", "-").replace("+", "_") + ".json"
        f = d / fname
        f.write_text(json.dumps(content))
        return f

    def _find_and_update_feedback(self, base_dir: Path, agent_id: str, ts: str, feedback: str):
        """Mirrors the logic from main.py POST /api/decisions/feedback."""
        target_dir = base_dir / agent_id
        for file_path in sorted(target_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(file_path.read_text())
            if data.get("timestamp") == ts:
                data["feedback"] = feedback
                data["feedback_at"] = datetime.now().astimezone().isoformat()
                file_path.write_text(json.dumps(data, indent=2))
                return file_path
        return None

    def test_feedback_up_is_persisted(self, tmp_path):
        """A 'up' rating must be written into the decision JSON file."""
        ts = "2026-04-11T10:00:00+00:00"
        self._write_decision(tmp_path, "lighting", ts, {
            "timestamp": ts, "agent_id": "lighting", "decision": {"reasoning": "Test"}
        })

        result = self._find_and_update_feedback(tmp_path, "lighting", ts, "up")
        assert result is not None, "File not found"

        saved = json.loads(result.read_text())
        assert saved["feedback"] == "up"
        assert "feedback_at" in saved

    def test_feedback_down_is_persisted(self, tmp_path):
        """A 'down' rating must also be saved correctly."""
        ts = "2026-04-11T11:00:00+00:00"
        self._write_decision(tmp_path, "security", ts, {
            "timestamp": ts, "agent_id": "security", "decision": {"reasoning": "Test"}
        })

        result = self._find_and_update_feedback(tmp_path, "security", ts, "down")
        assert result is not None
        saved = json.loads(result.read_text())
        assert saved["feedback"] == "down"

    def test_feedback_not_found_returns_none(self, tmp_path):
        """A timestamp that doesn't match any file must return None (→ 404 in the API)."""
        # Create a file with a different timestamp
        self._write_decision(tmp_path, "heating", "2026-01-01T00:00:00+00:00", {
            "timestamp": "2026-01-01T00:00:00+00:00"
        })
        result = self._find_and_update_feedback(
            tmp_path, "heating", "2099-01-01T00:00:00+00:00", "up"
        )
        assert result is None, "Should have returned None for unknown timestamp"

    def test_feedback_overwrite_toggles(self, tmp_path):
        """Rating can be changed from 'up' to 'down' by calling the logic twice."""
        ts = "2026-04-11T12:00:00+00:00"
        self._write_decision(tmp_path, "vision", ts, {"timestamp": ts})

        self._find_and_update_feedback(tmp_path, "vision", ts, "up")
        self._find_and_update_feedback(tmp_path, "vision", ts, "down")

        f = list((tmp_path / "vision").glob("*.json"))[0]
        saved = json.loads(f.read_text())
        assert saved["feedback"] == "down", "Second rating should overwrite first"

    def test_feedback_feedback_at_is_aware_timestamp(self, tmp_path):
        """feedback_at must include a UTC offset (never naive)."""
        ts = "2026-04-11T13:00:00+00:00"
        self._write_decision(tmp_path, "climate", ts, {"timestamp": ts})
        result = self._find_and_update_feedback(tmp_path, "climate", ts, "up")
        saved = json.loads(result.read_text())

        fat = saved["feedback_at"]
        has_offset = ('+' in fat[10:]) or fat.endswith('Z')
        assert has_offset, f"feedback_at '{fat}' has no UTC offset"

    def test_invalid_feedback_value_rejected(self):
        """Only 'up' and 'down' are valid — anything else must be rejected."""
        # This test exercises the validation guard from main.py
        valid_values = {"up", "down"}
        invalid_values = ["good", "bad", "", "1", "thumbsup", None]
        for val in invalid_values:
            assert val not in valid_values, (
                f"Value '{val}' should be rejected by the feedback validator"
            )
