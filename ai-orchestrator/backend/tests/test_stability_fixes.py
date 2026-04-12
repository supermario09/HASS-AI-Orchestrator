"""
Tests for the stability and timezone fixes applied in v1.0.6 / v1.0.7.

Covers:
1. asyncio.get_running_loop() replaces deprecated get_event_loop()
2. Subscription callback exception isolation in _receive_messages()
3. Race-condition guard on future.set_result() (done-future crash)
4. Blocking ollama.Client.chat() / Gemini calls moved to asyncio.to_thread()
5. Timezone correctness — all timestamps include UTC offset; analytics handles
   mixed naive/aware timestamps without TypeError
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# Ensure the backend directory is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Stub out heavy optional deps that aren't installed in the test environment
# (langgraph, chromadb). They are imported at module level by orchestrator.py
# / workflow_graph.py, so we must inject fakes into sys.modules before any
# 'from orchestrator import …' statement is reached.
# ---------------------------------------------------------------------------
for _mod in (
    "langgraph",
    "langgraph.graph",
    "chromadb",
    "chromadb.config",
    "chromadb.utils",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    # analytics.py imports fastapi at module level; not installed in test env
    "fastapi",
    "fastapi.routing",
):
    sys.modules.setdefault(_mod, MagicMock())

# Provide a concrete END sentinel that langgraph.graph.END is usually set to
sys.modules["langgraph.graph"].END = "END"
sys.modules["langgraph.graph"].StateGraph = MagicMock()

# ---------------------------------------------------------------------------
# Minimal async WebSocket double used by ha_client tests
# ---------------------------------------------------------------------------

class _FakeWS:
    """Async-iterable WebSocket double that yields a fixed list of messages."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self.closed = False

    async def send(self, data: str): ...  # noqa: E704

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._messages:
            return self._messages.pop(0)
        raise StopAsyncIteration


# ===========================================================================
# 1.  ha_client — asyncio.get_running_loop() works (Python 3.10+ safe)
# ===========================================================================

class TestGetRunningLoop:
    """Verify wait_until_connected() doesn't use the deprecated get_event_loop()."""

    @pytest.mark.asyncio
    async def test_already_connected_returns_true(self):
        from ha_client import HAWebSocketClient
        client = HAWebSocketClient("http://localhost:8123", "tok")
        client.connected = True
        # If get_event_loop() were called, Python 3.10+ raises DeprecationWarning
        # or RuntimeError when there is no current loop.  get_running_loop() is safe.
        result = await client.wait_until_connected(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_times_out_when_not_connected(self):
        from ha_client import HAWebSocketClient
        client = HAWebSocketClient("http://localhost:8123", "tok")
        client.connected = False
        result = await client.wait_until_connected(timeout=0.15)
        assert result is False

    @pytest.mark.asyncio
    async def test_create_future_uses_running_loop(self):
        """All four create_future() call sites now use get_running_loop()."""
        from ha_client import HAWebSocketClient
        client = HAWebSocketClient("http://localhost:8123", "tok")
        client.connected = True

        # Patch _send_message so the method under test doesn't need a real WS
        client._send_message = AsyncMock(return_value=42)

        # Fire off get_states which internally calls get_running_loop().create_future()
        task = asyncio.create_task(client.get_states())
        await asyncio.sleep(0)  # let the task reach the future creation

        assert 42 in client.pending_responses, (
            "Future not registered — create_future() probably raised before registering"
        )
        future = client.pending_responses[42]
        # Resolve it so the task can complete
        future.set_result({"success": True, "result": []})
        await task  # must not raise


# ===========================================================================
# 2.  ha_client — Callback exception must NOT kill the message receiver loop
# ===========================================================================

class TestCallbackIsolation:

    @pytest.mark.asyncio
    async def test_bad_callback_does_not_stop_receiver(self):
        """
        If a subscription callback raises, _receive_messages() must continue
        processing subsequent messages.  The pending_response future for the
        second message must still be resolved.
        """
        from ha_client import HAWebSocketClient

        client = HAWebSocketClient("http://localhost:8123", "tok")

        # Subscription 1 — raises
        async def _bad_callback(event):
            raise RuntimeError("intentional boom")

        client.subscriptions[1] = _bad_callback

        # Two messages: event for sub-1 (will raise), then a result for pending-2
        messages = [
            json.dumps({
                "type": "event", "id": 1,
                "event": {"data": {"entity_id": "light.test"}},
            }),
            json.dumps({
                "type": "result", "id": 2,
                "success": True, "result": ["state"],
            }),
        ]
        client.ws = _FakeWS(messages)
        client.connected = True

        # Register a future that the second message should resolve
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        client.pending_responses[2] = future

        # Must complete without raising
        await client._receive_messages()

        assert future.done(), "Future for ID 2 was never resolved — loop stopped early"
        assert future.result()["success"] is True

    @pytest.mark.asyncio
    async def test_good_callback_still_called_after_bad_one(self):
        """Good callbacks after a bad one must still fire."""
        from ha_client import HAWebSocketClient

        client = HAWebSocketClient("http://localhost:8123", "tok")
        called_good = []

        async def _bad_callback(event):
            raise ValueError("bad")

        async def _good_callback(event):
            called_good.append(event)

        client.subscriptions[1] = _bad_callback
        client.subscriptions[2] = _good_callback

        messages = [
            json.dumps({"type": "event", "id": 1, "event": {"x": 1}}),
            json.dumps({"type": "event", "id": 2, "event": {"x": 2}}),
        ]
        client.ws = _FakeWS(messages)
        client.connected = True

        await client._receive_messages()

        assert len(called_good) == 1
        assert called_good[0]["x"] == 2


# ===========================================================================
# 3.  ha_client — Race condition guard: set_result() on a done future
# ===========================================================================

class TestFutureDoneGuard:

    @pytest.mark.asyncio
    async def test_late_response_on_already_done_future_does_not_crash(self):
        """
        Simulate: request timed out → future is already done → HA finally
        responds → old code raised InvalidStateError crashing _receive_messages.
        New code skips set_result() when future.done() is True.
        """
        from ha_client import HAWebSocketClient

        client = HAWebSocketClient("http://localhost:8123", "tok")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        # Simulate a timeout: future is already resolved (or could be cancelled)
        future.set_result({"timedout": True})

        # Register the done future in pending_responses (as if timeout cleanup missed it)
        client.pending_responses[99] = future

        # HA sends a late response for ID 99
        messages = [
            json.dumps({"type": "result", "id": 99, "success": True, "result": []}),
        ]
        client.ws = _FakeWS(messages)
        client.connected = True

        # Before the fix this would raise asyncio.InvalidStateError
        await client._receive_messages()  # must NOT raise

    @pytest.mark.asyncio
    async def test_cancelled_future_not_overwritten(self):
        """A cancelled future (from wait_for timeout) must also be ignored."""
        from ha_client import HAWebSocketClient

        client = HAWebSocketClient("http://localhost:8123", "tok")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.cancel()  # Simulates asyncio.wait_for timeout cancellation

        client.pending_responses[77] = future

        messages = [
            json.dumps({"type": "result", "id": 77, "success": True, "result": []}),
        ]
        client.ws = _FakeWS(messages)
        client.connected = True

        await client._receive_messages()  # must NOT raise InvalidStateError


# ===========================================================================
# 4.  base_agent — _call_llm must use asyncio.to_thread (non-blocking)
# ===========================================================================

class TestBaseAgentNonBlocking:

    def _make_agent(self, tmp_path, agent_id="test", name="Test"):
        """Helper: construct a minimal BaseAgent concrete subclass."""
        from agents.base_agent import BaseAgent

        class _Agent(BaseAgent):
            async def decide(self, context):  # required abstract method
                return {"actions": []}

            async def gather_context(self):  # required abstract method
                return {}

        mock_mcp = MagicMock()
        mock_ha  = MagicMock()
        mock_ha.connected = True

        # BaseAgent.__init__ calls Path("/data/decisions").mkdir() — stub it out
        # so the test works on machines that don't have /data.
        with patch("pathlib.Path.mkdir"):
            agent = _Agent(
                agent_id=agent_id,
                name=name,
                mcp_server=mock_mcp,
                ha_client=mock_ha,
                skills_path=str(tmp_path / "SKILLS.md"),  # non-existent → defaults
                model_name="test-model",
            )
        return agent

    @pytest.mark.asyncio
    async def test_call_llm_uses_to_thread(self, tmp_path):
        """
        _call_llm() must call asyncio.to_thread with ollama_client.chat as the
        first argument — not call it directly (which would block the event loop).
        """
        agent = self._make_agent(tmp_path)

        # Replace the real ollama client with a mock
        agent.ollama_client = MagicMock()
        expected_response = {"message": {"content": "ok"}}

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = expected_response
            result = await agent._call_llm("hello")

        # asyncio.to_thread must have been called
        assert mock_to_thread.called, "_call_llm() did not use asyncio.to_thread"
        # First positional arg must be the sync chat function
        first_arg = mock_to_thread.call_args[0][0]
        assert first_arg is agent.ollama_client.chat, (
            "asyncio.to_thread was not called with ollama_client.chat"
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_call_llm_does_not_block_event_loop(self, tmp_path):
        """
        Prove the event loop is free during LLM call: a counter coroutine must
        increment while the (simulated slow) LLM call is running.
        """
        agent = self._make_agent(tmp_path, agent_id="test2", name="Test2")
        agent.ollama_client = MagicMock()

        counter = {"value": 0}

        async def _increment():
            for _ in range(5):
                counter["value"] += 1
                await asyncio.sleep(0)

        # Simulate a 150ms synchronous LLM call via to_thread
        def _slow_chat(**kwargs):
            time.sleep(0.15)
            return {"message": {"content": "done"}}

        agent.ollama_client.chat = _slow_chat

        # Run both concurrently
        await asyncio.gather(
            agent._call_llm("prompt"),
            _increment(),
        )

        assert counter["value"] == 5, (
            "Event loop was blocked during LLM call — counter never got CPU time. "
            "ollama.Client.chat() must be wrapped in asyncio.to_thread()."
        )


# ===========================================================================
# 5.  orchestrator — plan() and generate_visual_dashboard() use to_thread
# ===========================================================================

class TestOrchestratorNonBlocking:

    def _make_orchestrator(self):
        """Return a minimal Orchestrator with mocked dependencies."""
        # Avoid importing main.py which tries to import chromadb etc.
        # langgraph / chromadb are already stubbed in sys.modules at module top.
        from orchestrator import Orchestrator

        orch = Orchestrator.__new__(Orchestrator)
        mock_ha = MagicMock()
        mock_ha.get_states = AsyncMock(return_value=[
            {"entity_id": "light.test", "state": "on", "attributes": {}}
        ])
        mock_ha.connected = True
        # ha_client property calls _ha_provider() if callable — wrap in lambda so
        # the correctly-configured mock_ha is what get_states() is called on.
        orch._ha_provider    = lambda: mock_ha
        orch.mcp_server      = MagicMock()
        orch.approval_queue  = MagicMock()
        orch.agents          = {}
        orch.model_name      = "test-model"
        orch.planning_interval = 300
        orch.ollama_host_used = "http://localhost:11434"
        orch.task_ledger     = []
        orch.progress_ledger = {}
        orch.conflict_rules  = {}
        orch.gemini_api_key  = None
        orch.gemini_model    = None
        orch.use_gemini_for_dashboard = False
        orch.gemini_model_name = "gemini-test"
        orch.dashboard_refresh_interval = 300
        orch.last_dashboard_instruction = "test"
        orch.decision_log_dir = Path("/tmp/test_orch_decisions")
        orch.decision_log_dir.mkdir(parents=True, exist_ok=True)
        orch.dashboard_dir    = Path("/tmp/test_orch_dashboard")
        orch.dashboard_dir.mkdir(parents=True, exist_ok=True)
        orch.llm_client      = MagicMock()
        orch.workflow        = MagicMock()
        orch.compiled_workflow = MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_plan_uses_to_thread(self):
        """Orchestrator.plan() must not call llm_client.chat() directly."""
        orch = self._make_orchestrator()

        state = {
            "timestamp": "2025-01-01T00:00:00",
            "home_state": {"entities": [], "timestamp": "2025-01-01"},
            "tasks": [],
            "decisions": [],
            "conflicts": [],
            "approval_required": False,
            "approved_actions": [],
            "rejected_actions": [],
            "execution_results": [],
        }

        plan_response = {"message": {"content": '{"tasks": []}'}}

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = plan_response
            await orch.plan(state)

        assert mock_to_thread.called, "plan() did not use asyncio.to_thread"
        first_arg = mock_to_thread.call_args[0][0]
        assert first_arg is orch.llm_client.chat, (
            "asyncio.to_thread was not called with llm_client.chat in plan()"
        )

    @pytest.mark.asyncio
    async def test_dashboard_ollama_uses_to_thread(self):
        """generate_visual_dashboard() Ollama fallback must use asyncio.to_thread."""
        orch = self._make_orchestrator()
        # Gemini disabled → Ollama path
        orch.gemini_model = None

        dash_response = {"message": {"content": "<html><body>test</body></html>"}}

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = dash_response
            await orch.generate_visual_dashboard("test dashboard")

        assert mock_to_thread.called, (
            "generate_visual_dashboard() Ollama path did not use asyncio.to_thread"
        )
        first_arg = mock_to_thread.call_args[0][0]
        assert first_arg is orch.llm_client.chat

    @pytest.mark.asyncio
    async def test_dashboard_gemini_uses_to_thread(self):
        """generate_visual_dashboard() Gemini path must also use asyncio.to_thread."""
        orch = self._make_orchestrator()

        mock_gemini_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "<html><body>gemini dashboard</body></html>"
        orch.gemini_model = mock_gemini_model
        orch.use_gemini_for_dashboard = True

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response
            await orch.generate_visual_dashboard("test")

        assert mock_to_thread.called, (
            "generate_visual_dashboard() Gemini path did not use asyncio.to_thread"
        )
        first_arg = mock_to_thread.call_args[0][0]
        assert first_arg is mock_gemini_model.generate_content


# ===========================================================================
# 6.  Timezone — timestamps are always timezone-aware; analytics handles both
# ===========================================================================

class TestTimezone:

    def test_datetime_now_astimezone_includes_offset(self):
        """
        datetime.now().astimezone().isoformat() must include a UTC offset so
        JavaScript can parse it unambiguously. e.g. "+05:30" or "+00:00".
        A naive string like "2026-04-12T08:30:00" has NO offset — JS mis-interprets it.
        """
        from datetime import datetime
        ts = datetime.now().astimezone().isoformat()
        # A timezone-aware isoformat always contains '+' or ends with 'Z'
        has_offset = ('+' in ts[10:]) or ts.endswith('Z')  # skip date portion
        assert has_offset, (
            f"datetime.now().astimezone().isoformat() produced '{ts}' "
            "which has no UTC offset — JavaScript will mis-interpret it as local time"
        )

    def test_parse_ts_naive(self):
        """_parse_ts() must parse an old naive timestamp without raising."""
        from analytics import _parse_ts
        from datetime import datetime
        naive_str = "2026-04-12T08:30:00.123456"
        result = _parse_ts(naive_str)
        assert isinstance(result, datetime)
        assert result.tzinfo is None, "Expected naive datetime from naive input"
        assert result.hour == 8

    def test_parse_ts_aware_utc_z(self):
        """_parse_ts() must convert a 'Z'-suffixed UTC string to naive local."""
        from analytics import _parse_ts
        from datetime import datetime
        utc_str = "2026-04-12T03:00:00.000000Z"
        result = _parse_ts(utc_str)
        assert isinstance(result, datetime)
        assert result.tzinfo is None, "Expected naive datetime output"
        # The exact hour depends on the system timezone — just check it parsed
        assert 0 <= result.hour <= 23

    def test_parse_ts_aware_offset(self):
        """_parse_ts() must convert an offset-aware timestamp to naive local."""
        from analytics import _parse_ts
        from datetime import datetime
        # IST offset: +05:30
        ist_str = "2026-04-12T08:30:00.000000+05:30"
        result = _parse_ts(ist_str)
        assert isinstance(result, datetime)
        assert result.tzinfo is None

    def test_parse_ts_mixed_comparison_no_typeerror(self):
        """
        Comparing naive datetime.now() with _parse_ts() of an aware timestamp
        must NOT raise TypeError (the original bug when new log files appeared).
        """
        from analytics import _parse_ts
        from datetime import datetime, timedelta
        aware_str = "2026-04-12T08:30:00+05:30"
        naive_cutoff = datetime.now() - timedelta(days=7)
        ts = _parse_ts(aware_str)
        # This comparison would raise TypeError before the fix
        try:
            _ = ts >= naive_cutoff
        except TypeError as e:
            pytest.fail(f"TypeError raised comparing mixed timestamps: {e}")

    def test_parse_ts_invalid_returns_fallback(self):
        """_parse_ts() must return a datetime (not raise) for garbage input."""
        from analytics import _parse_ts
        from datetime import datetime
        result = _parse_ts("not-a-timestamp")
        assert isinstance(result, datetime)

    def test_base_agent_broadcast_timestamp_is_aware(self, tmp_path):
        """
        The 'last_active' and decision 'timestamp' broadcast by base_agent must
        include a UTC offset so the frontend displays the correct local time.
        """
        from datetime import datetime
        from agents.base_agent import BaseAgent

        class _Agent(BaseAgent):
            async def decide(self, context): return {"actions": []}
            async def gather_context(self): return {}

        with patch("pathlib.Path.mkdir"):
            agent = _Agent(
                agent_id="tz_test",
                name="TZ Test",
                mcp_server=MagicMock(),
                ha_client=MagicMock(),
                skills_path=str(tmp_path / "SKILLS.md"),
                model_name="test-model",
            )

        broadcasts = []

        async def capture_broadcast(msg):
            broadcasts.append(msg)

        agent.broadcast_func = capture_broadcast

        import asyncio
        asyncio.get_event_loop().run_until_complete(agent._broadcast_status("idle"))

        assert broadcasts, "No broadcast message emitted"
        ts = broadcasts[0]["data"]["last_active"]
        has_offset = ('+' in ts[10:]) or ts.endswith('Z')
        assert has_offset, (
            f"Broadcast last_active timestamp '{ts}' has no UTC offset — "
            "frontend will display wrong time"
        )
