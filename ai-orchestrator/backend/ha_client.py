"""
Home Assistant WebSocket client for real-time state subscriptions and service calls.
"""
import asyncio
import json
import logging
import base64
from typing import Dict, Callable, Optional, Any
from urllib.parse import urlparse

import websockets
import httpx

logger = logging.getLogger(__name__)


class HAWebSocketClient:
    """
    WebSocket client for Home Assistant integration.
    Handles authentication, state subscriptions, and service calls.

    Stability improvements over upstream:
    - Exponential back-off in reconnect loop (10 → 20 → 40 → 60 s cap)
    - Errors are logged instead of silently suppressed
    - Longer WebSocket ping timeout to survive slow/busy HA instances
    - Increased get_states() timeout for large entity registries (2000+ entities)
    - HTTP REST fallback for get_states() when WebSocket is sluggish
    - get_camera_snapshot() for vision integration
    """

    # Back-off configuration
    _RECONNECT_BASE_DELAY = 10     # seconds
    _RECONNECT_MAX_DELAY  = 60     # seconds

    def __init__(self, ha_url: str, token: str, supervisor_token: Optional[str] = None):
        """
        Initialize HA WebSocket client.

        Args:
            ha_url: Home Assistant URL (http/https)
            token: Token for WebSocket 'auth' packet (LLAT or Supervisor Token)
            supervisor_token: Token for Supervisor Proxy Headers (if different)
        """
        self.ha_url = ha_url.rstrip("/")
        self.token = token
        self.supervisor_token = supervisor_token or token
        self.connected = False
        self.ws = None
        self.message_id = 0
        self.subscriptions = {}
        self.pending_responses = {}

        # Convert HTTP URL to WebSocket URL
        parsed = urlparse(self.ha_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_url = f"{ws_scheme}://{parsed.netloc}{parsed.path}/api/websocket"
        self._closing = False
        self._reconnect_delay = self._RECONNECT_BASE_DELAY

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def disconnect(self):
        """Disconnect from Home Assistant"""
        self._closing = True
        self.connected = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        logger.info("📡 HA Client disconnected")

    async def connect(self):
        """Connect to Home Assistant WebSocket API and authenticate"""
        try:
            headers = {
                "Authorization": f"Bearer {self.supervisor_token}",
                "Content-Type": "application/json",
            }

            # Generous limits for large HA instances:
            #   max_size=10 MB   — handles huge entity registries
            #   ping_interval=30 — send keepalive every 30 s
            #   ping_timeout=120 — allow 2 min for the pong (fixes "keepalive ping timeout")
            self.ws = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                max_size=10 * 1024 * 1024,
                ping_interval=30,
                ping_timeout=120,
            )

            # Receive auth_required message
            auth_required = await self.ws.recv()
            auth_data = json.loads(auth_required)
            if auth_data["type"] != "auth_required":
                raise ValueError(f"Unexpected initial message: {auth_data}")

            # Send auth payload
            await self.ws.send(json.dumps({
                "type": "auth",
                "access_token": self.token
            }))

            # Receive auth result
            auth_result_raw = await self.ws.recv()
            auth_result = json.loads(auth_result_raw)
            if auth_result["type"] != "auth_ok":
                raise ValueError(f"Authentication failed: {auth_result}")

            self.connected = True
            self._reconnect_delay = self._RECONNECT_BASE_DELAY  # reset back-off on success

            # Start background message receiver
            asyncio.create_task(self._receive_messages())

        except Exception as e:
            if not self._closing:
                logger.error(f"❌ Failed to connect to HA WebSocket at {self.ws_url}: {repr(e)}")
            self.connected = False
            if self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass
                self.ws = None
            raise

    async def wait_until_connected(self, timeout: float = 30.0) -> bool:
        """Wait until connection is established or timeout occurs"""
        start = asyncio.get_event_loop().time()
        while not self.connected:
            if asyncio.get_event_loop().time() - start > timeout:
                return False
            await asyncio.sleep(0.5)
        return True

    # ------------------------------------------------------------------
    # Internal messaging
    # ------------------------------------------------------------------

    async def _send_message(self, message: Dict) -> int:
        """Send message to HA and return message ID"""
        if not self.ws or not self.connected:
            raise RuntimeError(
                f"Cannot send message ({message.get('type')}): Home Assistant not connected"
            )

        try:
            self.message_id += 1
            message["id"] = self.message_id
            await self.ws.send(json.dumps(message))
            return self.message_id
        except Exception as e:
            logger.error(f"❌ Error sending WebSocket message: {e}")
            self.connected = False
            return 0

    async def _receive_messages(self):
        """Continuously receive and process messages from HA"""
        try:
            async for message in self.ws:
                data = json.loads(message)

                # Handle subscription events
                if data["type"] == "event" and data.get("id") in self.subscriptions:
                    callback = self.subscriptions[data["id"]]
                    await callback(data["event"])

                # Handle command responses
                elif data.get("id") in self.pending_responses:
                    future = self.pending_responses.pop(data["id"])
                    future.set_result(data)

        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            if not self._closing:
                logger.warning("⚠️ HA WebSocket connection closed unexpectedly")
        except Exception as e:
            self.connected = False
            if not self._closing:
                logger.error(f"❌ Error in HA message receiver: {e}")
        finally:
            self.connected = False

    # ------------------------------------------------------------------
    # Reconnection loop with exponential back-off
    # ------------------------------------------------------------------

    async def run_reconnect_loop(self):
        """Infinite loop to maintain connection to Home Assistant with exponential back-off"""
        logger.info("🔄 HA Reconnection loop started")
        while not self._closing:
            if not self.connected:
                logger.info(f"📡 Attempting to connect to Home Assistant (next retry in {self._reconnect_delay}s if this fails)...")
                try:
                    await self.connect()
                    logger.info("✅ HA Reconnected successfully")
                    self._reconnect_delay = self._RECONNECT_BASE_DELAY  # reset on success
                except Exception as e:
                    # Log the failure so it's visible in the add-on logs
                    logger.warning(f"⚠️ HA connection attempt failed: {repr(e)} — retrying in {self._reconnect_delay}s")
                    await asyncio.sleep(self._reconnect_delay)
                    # Exponential back-off: double the delay, capped at max
                    self._reconnect_delay = min(self._reconnect_delay * 2, self._RECONNECT_MAX_DELAY)
                    continue

            await asyncio.sleep(10)

    # ------------------------------------------------------------------
    # HA API: States
    # ------------------------------------------------------------------

    async def get_states(self, entity_id: Optional[str] = None, timeout: float = 120.0):
        """
        Get current state of entities.

        Timeout increased to 120s to handle large entity registries (2000+ entities).
        Falls back to HTTP REST API if WebSocket is not connected.

        Args:
            entity_id: Specific entity ID, or None for all entities
            timeout:   Timeout in seconds (default: 120.0)

        Returns:
            Entity state dict, or list of states
        """
        if not self.connected:
            # REST fallback when WebSocket is down
            return await self._get_states_rest(entity_id)

        try:
            msg_id = await self._send_message({"type": "get_states"})

            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self.pending_responses[msg_id] = future
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                self.pending_responses.pop(msg_id, None)
                logger.warning("⚠️ get_states() WebSocket timeout — falling back to REST")
                return await self._get_states_rest(entity_id)

            if not result.get("success"):
                raise ValueError(f"Failed to get states: {result}")

            states = result["result"]

        except RuntimeError:
            # Not connected — fall back to REST
            return await self._get_states_rest(entity_id)

        if entity_id:
            for state in states:
                if state["entity_id"] == entity_id:
                    return state
            raise ValueError(f"Entity {entity_id} not found")

        return states

    async def _get_states_rest(self, entity_id: Optional[str] = None):
        """HTTP REST fallback for get_states() when WebSocket is unavailable"""
        headers = {"Authorization": f"Bearer {self.token}"}
        path = f"/api/states/{entity_id}" if entity_id else "/api/states"
        url = f"{self.ha_url}{path}"

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # HA API: Services
    # ------------------------------------------------------------------

    async def get_services(self) -> Dict:
        """Get all available services from Home Assistant."""
        msg_id = await self._send_message({"type": "get_services"})

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending_responses[msg_id] = future
        try:
            result = await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            self.pending_responses.pop(msg_id, None)
            raise TimeoutError("Timeout waiting for HA services")

        if not result.get("success"):
            raise ValueError(f"Failed to get services: {result}")

        return result["result"]

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Call a Home Assistant service.

        Args:
            domain:    Service domain (e.g. 'climate', 'light', 'tts')
            service:   Service name (e.g. 'set_temperature', 'turn_on', 'speak')
            entity_id: Target entity ID (optional — some services don't need one)
            **kwargs:  Additional service data

        Returns:
            Service call result
        """
        service_data = kwargs.copy()
        if entity_id:
            service_data["entity_id"] = entity_id

        msg_id = await self._send_message({
            "type": "call_service",
            "domain": domain,
            "service": service,
            "service_data": service_data,
        })

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending_responses[msg_id] = future
        result = await asyncio.wait_for(future, timeout=10.0)

        if not result.get("success"):
            raise ValueError(f"Service call failed: {result}")

        return result.get("result") or {}

    # ------------------------------------------------------------------
    # HA API: Subscriptions
    # ------------------------------------------------------------------

    async def subscribe_entities(
        self,
        entity_ids: list,
        callback: Callable[[Dict], Any],
    ) -> int:
        """Subscribe to state_changed events for specific entities."""
        msg_id = await self._send_message({
            "type": "subscribe_events",
            "event_type": "state_changed",
        })

        async def filtered_callback(event):
            if event["data"]["entity_id"] in entity_ids:
                await callback(event)

        self.subscriptions[msg_id] = filtered_callback

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending_responses[msg_id] = future
        result = await asyncio.wait_for(future, timeout=10.0)

        if not result.get("success"):
            raise ValueError(f"Subscription failed: {result}")

        return msg_id

    # ------------------------------------------------------------------
    # HA API: Climate helpers
    # ------------------------------------------------------------------

    async def get_climate_state(self, entity_id: str) -> Dict:
        """Get climate entity state with temperature and HVAC info."""
        state = await self.get_states(entity_id)
        return {
            "entity_id": entity_id,
            "state": state["state"],
            "current_temperature": state["attributes"].get("current_temperature"),
            "target_temperature": state["attributes"].get("temperature"),
            "hvac_mode": state["attributes"].get("hvac_mode"),
            "preset_mode": state["attributes"].get("preset_mode"),
            "attributes": state["attributes"],
        }

    # ------------------------------------------------------------------
    # HA API: Camera snapshot (Vision integration)
    # ------------------------------------------------------------------

    async def get_camera_snapshot(self, entity_id: str, timeout: float = 15.0) -> bytes:
        """
        Capture a JPEG snapshot from a Home Assistant camera entity.

        Uses the HA REST API endpoint:  GET /api/camera_proxy/{entity_id}

        Args:
            entity_id: Camera entity (e.g. 'camera.front_entrance_doorbell')
            timeout:   HTTP timeout in seconds

        Returns:
            Raw JPEG bytes that can be sent to Gemini Vision or saved to disk.

        Raises:
            httpx.HTTPStatusError: If HA returns a non-2xx status
            RuntimeError: If the response is empty
        """
        url = f"{self.ha_url}/api/camera_proxy/{entity_id}"
        headers = {"Authorization": f"Bearer {self.token}"}

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        image_bytes = resp.content
        if not image_bytes:
            raise RuntimeError(f"Empty camera snapshot from {entity_id}")

        logger.debug(f"📸 Camera snapshot captured: {entity_id} ({len(image_bytes)} bytes)")
        return image_bytes
