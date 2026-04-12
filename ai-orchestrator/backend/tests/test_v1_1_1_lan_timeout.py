"""
Tests for v1.1.1 LAN-timeout fix.

Covers:
1. BaseAgent.ollama_client is constructed with httpx.Timeout(read=120)
2. BaseAgent._call_llm passes num_ctx=4096 and num_predict=1000
3. BaseAgent._call_llm retry delays are 3s and 8s (tuned for LAN blips)
4. VisionAgent._ollama_client is also constructed with httpx.Timeout(read=120)
5. VisionAgent._analyze_with_ollama_text passes num_ctx=4096 to Ollama
6. VisionAgent retry delays are also [3, 8]
7. BaseAgent._call_llm retries on empty response (integration of semaphore + timeout)
8. BaseAgent._call_llm returns ERROR string after all retries exhausted
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# ── stubs for heavy optional deps ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

for _mod in (
    "langgraph", "langgraph.graph", "chromadb", "chromadb.config",
    "chromadb.utils", "langgraph.checkpoint", "langgraph.checkpoint.memory",
    "fastapi", "fastapi.routing", "google.generativeai", "PIL", "PIL.Image",
):
    sys.modules.setdefault(_mod, MagicMock())


# ─────────────────────────────────────────────────────────────────────────────
# Utility: replace asyncio.to_thread with a direct call (no thread pool needed)
# ─────────────────────────────────────────────────────────────────────────────

async def _fake_to_thread(func, *args, **kwargs):
    """Call func synchronously — avoids thread pool in tests."""
    return func(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_base_agent(model_name="mistral:7b-instruct"):
    """Instantiate a minimal concrete BaseAgent for testing."""
    from agents.base_agent import BaseAgent

    class ConcreteAgent(BaseAgent):
        async def decide(self, ctx):
            return {"actions": []}

        async def gather_context(self):
            return {}

    mcp = MagicMock()
    mcp.dry_run = False

    with patch("agents.base_agent.ollama.Client") as mock_client_cls, \
         patch("pathlib.Path.mkdir"):
        mock_client_cls.return_value = MagicMock()
        agent = ConcreteAgent(
            agent_id="test",
            name="Test",
            mcp_server=mcp,
            ha_client=MagicMock(),
            skills_path="/nonexistent/SKILLS.md",
            model_name=model_name,
        )
        return agent, mock_client_cls


def _make_vision_agent():
    """Instantiate a minimal VisionAgent for testing."""
    from agents.vision_agent import VisionAgent

    with patch("agents.vision_agent.ollama.Client") as mock_client_cls, \
         patch("pathlib.Path.mkdir"):
        mock_client_cls.return_value = MagicMock()
        agent = VisionAgent(
            agent_id="vision",
            name="Vision",
            instruction="Monitor cameras",
            mcp_server=MagicMock(),
            ha_client=MagicMock(),
            entities=["camera.front"],
            gemini_api_key="",
            vision_enabled=False,
        )
        return agent, mock_client_cls


# ─────────────────────────────────────────────────────────────────────────────
# 1. BaseAgent — httpx.Timeout(read=120) used for ollama.Client
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseAgentOllamaClientTimeout:

    def test_ollama_client_receives_httpx_timeout(self):
        """ollama.Client must be constructed with an explicit httpx.Timeout."""
        import httpx
        _, mock_client_cls = _make_base_agent()

        assert mock_client_cls.called, "ollama.Client was not instantiated"
        _, kwargs = mock_client_cls.call_args
        timeout = kwargs.get("timeout")
        assert timeout is not None, "timeout kwarg missing from ollama.Client()"
        assert isinstance(timeout, httpx.Timeout), (
            f"Expected httpx.Timeout, got {type(timeout)}"
        )

    def test_ollama_client_read_timeout_is_120(self):
        """The read timeout must be 120s to survive LAN latency to Ollama."""
        _, mock_client_cls = _make_base_agent()
        _, kwargs = mock_client_cls.call_args
        timeout = kwargs["timeout"]
        assert timeout.read == 120.0, (
            f"read timeout should be 120s for M4 Mac Mini on LAN, got {timeout.read}"
        )

    def test_ollama_client_connect_timeout_is_10(self):
        """Connect timeout should be 10s (fast failure if Ollama is unreachable)."""
        _, mock_client_cls = _make_base_agent()
        _, kwargs = mock_client_cls.call_args
        timeout = kwargs["timeout"]
        assert timeout.connect == 10.0, (
            f"connect timeout should be 10s, got {timeout.connect}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. BaseAgent._call_llm — generation parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseAgentCallLlmParams:

    @pytest.mark.asyncio
    async def test_call_llm_uses_num_ctx_4096(self):
        """_call_llm must pass num_ctx=4096 (M4 Mac Mini has headroom)."""
        agent, _ = _make_base_agent()
        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.return_value = {"message": {"content": "hello"}}

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("asyncio.sleep"):
            await agent._call_llm("test prompt")

        _, kwargs = agent.ollama_client.chat.call_args
        options = kwargs.get("options", {})
        assert options.get("num_ctx") == 4096, (
            f"num_ctx should be 4096, got {options.get('num_ctx')}"
        )

    @pytest.mark.asyncio
    async def test_call_llm_uses_num_predict_1000(self):
        """_call_llm must pass num_predict=1000 (not capped low for fast hardware)."""
        agent, _ = _make_base_agent()
        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.return_value = {"message": {"content": "hello"}}

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("asyncio.sleep"):
            await agent._call_llm("test prompt", max_tokens=1000)

        _, kwargs = agent.ollama_client.chat.call_args
        options = kwargs.get("options", {})
        assert options.get("num_predict") == 1000, (
            f"num_predict should be 1000 for M4 Mac Mini, got {options.get('num_predict')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. BaseAgent._call_llm — retry delays are [3, 8] (LAN-tuned, not Pi-tuned)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseAgentRetryDelays:

    @pytest.mark.asyncio
    async def test_retry_delays_are_3_and_8_seconds(self):
        """
        When the LLM returns an empty response, delays between retries
        should be 3s then 8s — not 5s/10s (those were for slow Pi hardware).
        """
        agent, _ = _make_base_agent()
        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.side_effect = [
            {"message": {"content": ""}},              # attempt 0: empty
            {"message": {"content": ""}},              # attempt 1: empty
            {"message": {"content": "good response"}}, # attempt 2: ok
        ]

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.base_agent.asyncio.sleep", side_effect=fake_sleep):
            result = await agent._call_llm("prompt")

        assert result == "good response"
        assert sleep_calls == [3, 8], (
            f"Expected retry delays [3, 8] seconds, got {sleep_calls}"
        )

    @pytest.mark.asyncio
    async def test_exhausted_retries_returns_error_string(self):
        """After 3 failed attempts, _call_llm should return an ERROR: string."""
        agent, _ = _make_base_agent()
        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.return_value = {"message": {"content": ""}}

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.base_agent.asyncio.sleep"):
            result = await agent._call_llm("prompt")

        assert result.startswith("ERROR:"), f"Expected ERROR: prefix, got: {result!r}"

    @pytest.mark.asyncio
    async def test_exception_retries_with_correct_delays(self):
        """Exceptions (network errors) also trigger [3, 8] delays."""
        agent, _ = _make_base_agent()
        agent.ollama_client = MagicMock()
        agent.ollama_client.chat.side_effect = [
            Exception("connection reset"),
            Exception("timeout"),
            {"message": {"content": "recovered"}},
        ]

        sleep_calls = []

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        with patch("agents.base_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.base_agent.asyncio.sleep", side_effect=fake_sleep):
            result = await agent._call_llm("prompt")

        assert result == "recovered"
        assert sleep_calls == [3, 8], f"Expected [3, 8], got {sleep_calls}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. VisionAgent — httpx.Timeout(read=120) on _ollama_client
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionAgentOllamaClientTimeout:

    def test_vision_ollama_client_receives_httpx_timeout(self):
        """VisionAgent._ollama_client must also have an explicit httpx.Timeout."""
        import httpx
        _, mock_client_cls = _make_vision_agent()

        assert mock_client_cls.called, "ollama.Client was not instantiated in VisionAgent"
        _, kwargs = mock_client_cls.call_args
        timeout = kwargs.get("timeout")
        assert timeout is not None, "timeout kwarg missing from VisionAgent's ollama.Client()"
        assert isinstance(timeout, httpx.Timeout)

    def test_vision_ollama_client_read_timeout_is_120(self):
        """VisionAgent read timeout must be 120s for LAN Ollama calls."""
        _, mock_client_cls = _make_vision_agent()
        _, kwargs = mock_client_cls.call_args
        timeout = kwargs["timeout"]
        assert timeout.read == 120.0, (
            f"VisionAgent read timeout should be 120s, got {timeout.read}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5 & 6. VisionAgent._analyze_with_ollama_text — num_ctx=4096 + retry delays
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionAgentOllamaTextParams:

    @pytest.mark.asyncio
    async def test_analyze_with_ollama_text_uses_num_ctx_4096(self):
        """_analyze_with_ollama_text must pass num_ctx=4096."""
        agent, _ = _make_vision_agent()

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value=[])
        agent._ha_provider = mock_ha

        agent._ollama_client = MagicMock()
        agent._ollama_client.chat.return_value = {
            "message": {"content": "All clear, no activity detected."}
        }

        with patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.vision_agent.asyncio.sleep"):
            result = await agent._analyze_with_ollama_text("camera.front")

        assert result is not None
        _, kwargs = agent._ollama_client.chat.call_args
        options = kwargs.get("options", {})
        assert options.get("num_ctx") == 4096, (
            f"VisionAgent num_ctx should be 4096, got {options.get('num_ctx')}"
        )

    @pytest.mark.asyncio
    async def test_analyze_with_ollama_text_retry_delays_are_3_and_8(self):
        """VisionAgent._analyze_with_ollama_text first retry delay must be 3s."""
        agent, _ = _make_vision_agent()

        mock_ha = MagicMock()
        mock_ha.connected = True
        mock_ha.get_states = AsyncMock(return_value=[])
        agent._ha_provider = mock_ha

        agent._ollama_client = MagicMock()
        agent._ollama_client.chat.side_effect = [
            {"message": {"content": ""}},           # attempt 0: empty
            {"message": {"content": "all clear"}},  # attempt 1: ok
        ]

        sleep_calls = []

        async def fake_sleep(d):
            sleep_calls.append(d)

        with patch("agents.vision_agent.asyncio.to_thread", side_effect=_fake_to_thread), \
             patch("agents.vision_agent.asyncio.sleep", side_effect=fake_sleep):
            result = await agent._analyze_with_ollama_text("camera.front")

        assert result == "all clear"
        assert sleep_calls[0] == 3, (
            f"First retry delay should be 3s (LAN blip), got {sleep_calls[0]}s"
        )
