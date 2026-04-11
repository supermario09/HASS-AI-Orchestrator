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
        self.model_name = "gemini-vision"
        self._vision_enabled = vision_enabled

        # Gemini setup
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self._vision_model = None
        if not vision_enabled:
            logger.info("VisionAgent: use_gemini_for_vision=false — vision analysis disabled")
        elif _GENAI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self._vision_model = genai.GenerativeModel(gemini_model_name)
                logger.info(f"✅ VisionAgent: Gemini {gemini_model_name} configured")
            except Exception as e:
                logger.warning(f"⚠️ VisionAgent: Gemini init failed: {e}")
        elif not _GENAI_AVAILABLE:
            logger.warning("VisionAgent: google-generativeai not installed — vision disabled")
        elif not api_key:
            logger.warning("VisionAgent: GEMINI_API_KEY not set — vision disabled")

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
        Capture and analyze a single camera snapshot.

        Returns the Gemini analysis string, or None on failure.
        """
        if not self._vision_model:
            return None

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
            response = self._vision_model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            # 429 rate limit — back off without logging as an error
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                logger.warning(
                    f"VisionAgent: Gemini rate limit (429) for {entity_id}. "
                    "Backing off 60s. Consider increasing decision_interval or "
                    "disabling use_gemini_for_vision."
                )
                await asyncio.sleep(60)
            else:
                logger.error(f"VisionAgent: Gemini call failed for {entity_id}: {e}")
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
            logger.warning(
                "VisionAgent: No Gemini model available. "
                "Set GEMINI_API_KEY and ensure google-generativeai is installed."
            )
            # Keep the loop alive so it can be re-checked if config changes
            while True:
                await asyncio.sleep(60)

        while True:
            self.status = "deciding"
            observations: List[Dict] = []

            for entity_id in self.entities:
                analysis = await self.analyze_camera(entity_id)
                if analysis:
                    obs = {
                        "camera": entity_id,
                        "analysis": analysis,
                        "timestamp": datetime.now().isoformat(),
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
                        "timestamp": datetime.now().isoformat(),
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
                    "timestamp": datetime.now().isoformat(),
                },
            })
        except Exception:
            pass
