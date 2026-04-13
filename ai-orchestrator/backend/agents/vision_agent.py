"""
VisionAgent — monitors Home Assistant cameras using Gemini Vision AI.

Captures JPEG snapshots via the HA REST API and sends them to Gemini 1.5 Pro
for analysis.  When activity is detected the agent can:
  - Announce findings via TTS (speak_tts MCP tool)
  - Log decisions to the standard decision log
  - Broadcast status updates to the dashboard WebSocket

Cameras configured in agents.yaml under the 'vision' agent entry.

This agent works even when Ollama is unavailable because it routes vision
inference through the Google Gemini API rather than a local LLM.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ollama

logger = logging.getLogger(__name__)

# Guard Gemini import
try:
    import google.generativeai as genai
    import PIL.Image
    import io as _io
    _GENAI_AVAILABLE = True
except Exception:
    genai = None  # type: ignore[assignment]
    _GENAI_AVAILABLE = False

# Re-use the same global LLM semaphore as BaseAgent so vision Ollama calls
# are also serialised and don't pile onto the model alongside other agents.
from agents.base_agent import _get_llm_semaphore


class VisionAgent:
    """
    Autonomous camera-monitoring agent powered by Gemini Vision.

    Runs a decision loop that:
    1. Captures a snapshot from each configured camera
    2. Sends it to Gemini Vision with a configurable prompt
    3. Speaks alerts via TTS for significant findings
    4. Logs all observations to /data/decisions/vision/
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        instruction: str,
        mcp_server: Any,
        ha_client: Any,
        entities: List[str],           # Camera entity IDs
        gemini_api_key: Optional[str] = None,
        gemini_model_name: str = "gemini-robotics-er-1.5-preview",
        decision_interval: int = 60,
        default_media_player: str = "media_player.kitchen_display",
        broadcast_func: Optional[Any] = None,
        vision_enabled: bool = False,
        ollama_model: str = "mistral:7b-instruct",
        ollama_host: str = "http://localhost:11434",
    ):
        self.agent_id = agent_id
        self.name = name
        self.instruction = instruction
        self._mcp_server = mcp_server
        self._ha_provider = ha_client
        self.entities = entities        # camera entity IDs
        self.decision_interval = decision_interval
        self.default_media_player = default_media_player
        self._broadcast_func = broadcast_func
        self.status = "idle"
        self._vision_enabled = vision_enabled
        self._ollama_model = ollama_model
        import httpx as _httpx
        self._ollama_client = ollama.Client(
            host=os.getenv("OLLAMA_HOST", ollama_host),
            timeout=_httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )

        # Gemini setup — preferred when api_key is available
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self._vision_model = None
        if not vision_enabled:
            logger.info("VisionAgent: use_gemini_for_vision=false — will use Ollama text fallback")
        elif _GENAI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self._vision_model = genai.GenerativeModel(gemini_model_name)
                logger.info(f"✅ VisionAgent: Gemini {gemini_model_name} configured")
            except Exception as e:
                logger.warning(f"⚠️ VisionAgent: Gemini init failed: {e} — using Ollama fallback")
        elif not _GENAI_AVAILABLE:
            logger.warning("VisionAgent: google-generativeai not installed — using Ollama text fallback")
        elif not api_key:
            logger.warning("VisionAgent: GEMINI_API_KEY not set — using Ollama text fallback")

        # Determine display model name for UI
        self.model_name = f"gemini-vision" if self._vision_model else f"ollama:{ollama_model}"

        # Decision log directory
        self._log_dir = Path("/data/decisions/vision")
        if not (Path("/data").exists() and os.access("/data", os.W_OK)):
            self._log_dir = Path(__file__).parent.parent.parent / "data" / "decisions" / "vision"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ha_client(self):
        if callable(self._ha_provider):
            return self._ha_provider()
        return self._ha_provider

    @property
    def mcp_server(self):
        if callable(self._mcp_server):
            return self._mcp_server()
        return self._mcp_server

    # ------------------------------------------------------------------
    # Compatibility helpers (used by main.py loops)
    # ------------------------------------------------------------------

    def get_last_decision_file(self) -> Optional[Path]:
        """Return the most recent decision log file, or None."""
        files = sorted(self._log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None

    # ------------------------------------------------------------------
    # Vision analysis
    # ------------------------------------------------------------------

    async def analyze_camera(self, entity_id: str) -> Optional[str]:
        """
        Analyze a camera.  Uses Gemini Vision when configured, otherwise falls
        back to Ollama text-based analysis of the camera entity's HA state.
        """
        if self._vision_model:
            return await self._analyze_with_gemini(entity_id)
        else:
            return await self._analyze_with_ollama_text(entity_id)

    async def _analyze_with_gemini(self, entity_id: str) -> Optional[str]:
        """Capture JPEG snapshot and send to Gemini Vision."""
        client = self.ha_client
        if not client:
            return None

        try:
            image_bytes = await client.get_camera_snapshot(entity_id, timeout=15.0)
        except Exception as e:
            logger.warning(f"VisionAgent: snapshot failed for {entity_id}: {e}")
            return None

        try:
            image = PIL.Image.open(_io.BytesIO(image_bytes))
            prompt = (
                f"Camera: {entity_id}\n"
                f"Agent instruction: {self.instruction}\n\n"
                "Analyze this camera image. Answer:\n"
                "1. What do you see? (brief, 1-2 sentences)\n"
                "2. Is there any security concern, person, vehicle, or unusual activity?\n"
                "3. Should I alert the household? (yes/no and why)\n"
                "Keep your response under 100 words."
            )
            async with _get_llm_semaphore():
                response = await asyncio.to_thread(
                    self._vision_model.generate_content,
                    [prompt, image],
                )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                logger.warning(
                    f"VisionAgent: Gemini rate limit (429) for {entity_id}. Backing off 60s."
                )
                await asyncio.sleep(60)
            else:
                logger.error(f"VisionAgent: Gemini call failed for {entity_id}: {e}")
            return None

    async def _analyze_with_ollama_text(self, entity_id: str) -> Optional[str]:
        """
        Ollama text-based fallback when Gemini is unavailable.

        Fetches the camera entity's state and nearby sensor attributes from HA,
        then asks the configured Ollama model to reason about security/activity.
        No image is involved — this works with any text model (mistral, deepseek…).
        """
        client = self.ha_client
        if not client or not client.connected:
            return None

        # Gather camera state + related sensor context
        context_lines = [f"Camera entity: {entity_id}"]
        try:
            cam_state = await client.get_states(entity_id)
            if cam_state:
                state_val = cam_state.get("state", "unknown")
                attrs = cam_state.get("attributes", {})
                context_lines.append(f"Camera state: {state_val}")
                # Include any motion/person detection attributes
                for k, v in attrs.items():
                    if any(kw in k.lower() for kw in ("motion", "person", "detect", "activity", "trigger")):
                        context_lines.append(f"  {k}: {v}")
        except Exception:
            context_lines.append("Camera state: unavailable")

        # Also pull states of any related binary_sensor / sensor entities
        # (cameras often have companion motion sensors with the same area prefix)
        area_prefix = entity_id.split(".")[1].split("_")[0] if "." in entity_id else ""
        try:
            all_states = await client.get_states()
            related = [
                s for s in (all_states or [])
                if s.get("entity_id", "").split(".")[0] in ("binary_sensor", "sensor")
                and area_prefix
                and area_prefix in s.get("entity_id", "")
            ][:5]
            for s in related:
                eid2 = s.get("entity_id", "")
                context_lines.append(f"Related sensor {eid2}: {s.get('state', '?')}")
        except Exception:
            pass

        context = "\n".join(context_lines)
        prompt = (
            f"You are a security monitor. Here is the current state of a camera and nearby sensors:\n\n"
            f"{context}\n\n"
            f"Agent instruction: {self.instruction}\n\n"
            "Based on this sensor data (not an image):\n"
            "1. What can you infer is happening? (1-2 sentences)\n"
            "2. Is there any security concern or unusual activity?\n"
            "3. Should I alert the household? (yes/no and brief reason)\n"
            "Keep your response under 80 words."
        )

        import re
        _MAX_ATTEMPTS = 3
        _RETRY_DELAYS = [3, 8]   # short — LAN network blips, not hardware slowness

        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    f"VisionAgent: Ollama attempt {attempt} empty/failed for {entity_id}. "
                    f"Retrying in {delay}s…"
                )
                await asyncio.sleep(delay)

            try:
                async with _get_llm_semaphore():
                    response = await asyncio.to_thread(
                        self._ollama_client.chat,
                        model=self._ollama_model,
                        messages=[{"role": "user", "content": prompt}],
                        options={
                            "temperature": 0.3,
                            "num_predict": 200,
                            "num_ctx": 2048,       # vision prompts fit within 2k easily
                            "think": False,
                            "repeat_penalty": 1.0,
                        },
                        keep_alive=-1,             # never unload; avoids reload cost
                        stream=False,
                    )
                content = response["message"]["content"]
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                content = content.strip()
                if content:
                    return content
            except Exception as e:
                logger.error(f"VisionAgent: Ollama attempt {attempt + 1} failed for {entity_id}: {e}")

        logger.error(f"VisionAgent: Ollama fallback gave up after {_MAX_ATTEMPTS} attempts for {entity_id}")
        return None

    # ------------------------------------------------------------------
    # TTS announcement
    # ------------------------------------------------------------------

    async def _announce(self, message: str):
        """Speak a message via the speak_tts MCP tool."""
        try:
            await self.mcp_server.execute_tool(
                "speak_tts",
                {"message": message, "media_player": self.default_media_player},
                agent_id=self.agent_id,
            )
        except Exception as e:
            logger.error(f"VisionAgent: TTS announcement failed: {e}")

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    async def run_decision_loop(self):
        """Main autonomous vision monitoring loop."""
        logger.info(
            f"👁️ VisionAgent '{self.name}' started — "
            f"monitoring {len(self.entities)} camera(s) every {self.decision_interval}s"
        )

        if not self._vision_enabled:
            logger.info(
                "VisionAgent: Vision analysis disabled (use_gemini_for_vision=false). "
                "Enable it in the add-on options to start camera monitoring."
            )
            # Keep the loop alive so status is visible; wake up occasionally to recheck
            while True:
                await asyncio.sleep(300)

        if not self._vision_model:
            logger.info(
                "VisionAgent: Gemini Vision unavailable — using Ollama text-based fallback for all cameras."
            )
            # Fall through to the main loop; analyze_camera() will use _analyze_with_ollama_text()

        while True:
            self.status = "deciding"
            observations: List[Dict] = []

            for entity_id in self.entities:
                analysis = await self.analyze_camera(entity_id)
                if analysis:
                    obs = {
                        "camera": entity_id,
                        "analysis": analysis,
                        "timestamp": datetime.now().astimezone().isoformat(),
                    }
                    observations.append(obs)
                    logger.info(f"📷 [{entity_id}]: {analysis[:120]}")

                    # Alert on significant findings (simple keyword heuristic)
                    alert_keywords = [
                        "person", "people", "someone", "delivery", "vehicle",
                        "car", "alert", "unusual", "suspicious", "package",
                        "open", "broken", "smoke", "fire"
                    ]
                    if any(kw in analysis.lower() for kw in alert_keywords):
                        # Summarise to ≤160 chars for TTS
                        summary = analysis.split(".")[0][:160]
                        await self._announce(f"Camera alert: {summary}")

            # Log all observations
            if observations:
                self._save_decision(observations)
                await self._broadcast(observations)

            self.status = "idle"
            await asyncio.sleep(self.decision_interval)

    # ------------------------------------------------------------------
    # Logging / broadcasting
    # ------------------------------------------------------------------

    def _save_decision(self, observations: List[Dict]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = self._log_dir / f"{ts}.json"
        try:
            with open(log_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "agent_id": self.agent_id,
                        "observations": observations,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"VisionAgent: failed to save log: {e}")

    async def _broadcast(self, observations: List[Dict]):
        if not self._broadcast_func:
            return
        try:
            await self._broadcast_func({
                "type": "agent_update",
                "data": {
                    "agent_id": self.agent_id,
                    "name": self.name,
                    "status": self.status,
                    "observations": observations,
                    "timestamp": datetime.now().astimezone().isoformat(),
                },
            })
        except Exception:
            pass
