# Changelog
<br>

## [1.2.3] - 2026-04-13
### Fixed
- **Restore `num_predict=1000`**: deepseek-r1:8b silently ignores `think: False` on some Ollama versions and generates a `<think>...</think>` block before every response. With `num_predict=500` (set in v1.2.2), the think block alone consumed the full budget — after stripping it, content was empty → "LLM returned empty response" on every call. Restoring to 1000 gives enough room for a think block (~300 tokens) plus the full JSON response.
- **Fix warmup prompt**: the `"ping"` warmup with `max_tokens=3` hit the same problem. Changed to an explicit `'Reply with "ready" and nothing else.'` prompt with `max_tokens=200` so the warmup reliably succeeds.
<br>
<br>

## [1.2.2] - 2026-04-13
### Fixed / Improved
- **`keep_alive=-1` on all Ollama calls**: model is now kept loaded in Ollama memory indefinitely. Without this, Ollama unloads the model after its default 5-minute timeout. With 300s polling intervals this hit every other decision cycle, adding 5-15s of model-reload overhead to each call.
- **`num_ctx` reduced from 4096 → 2048**: halves the KV-cache allocation. Decision prompts are 600-1200 tokens — the 4096 window was never needed. Smaller KV cache means faster prefill and less VRAM pressure.
- **Default `temperature` lowered from 0.7 → 0.3**: agents produce structured JSON decisions. High temperature adds noise without benefit. Lower entropy speeds up sampling slightly and reduces JSON parse errors.
- **Default `num_predict` reduced from 1000 → 500**: JSON decisions are 50-300 tokens. Capping earlier terminates generation sooner without truncating real responses.
- **`repeat_penalty=1.0`**: disables the repeat-penalty computation pass on every token. Useful for prose; wasted overhead for short structured JSON.
- **Model warmup before first decision**: `run_decision_loop` fires a minimal `_call_llm("ping", max_tokens=3)` after HA connects. This pre-loads the model into Ollama's memory so the first real decision doesn't pay the load penalty.
- **Services cache in UniversalAgent**: `get_services()` was called on every `decide()` — a network round-trip to HA every 300s. Services almost never change (only on integration reload). Now cached for 10 minutes.
- **Tests** (12 new): `keep_alive=-1` captured from chat() call, `num_ctx=2048` in both base and vision agent, temperature/max_tokens defaults, `repeat_penalty=1.0`, warmup fires before gather_context, services cache hit on second call, cache expiry triggers re-fetch (134 total).
<br>
<br>

## [1.2.1] - 2026-04-13
### Fixed
- **Reverted event-driven agent wake**: removed `event_driven`, `_trigger_event`, `_on_entity_changed`, and `asyncio.wait_for` wake logic from `base_agent.py`. Event-driven triggering increased Ollama call frequency and made timeouts worse — HA native automations are better suited to reactive triggering. Agents now use simple polling only.
- **Reverted per-model semaphores**: replaced the per-model `_LLM_SEMAPHORES` dict with the original single `_LLM_SEMAPHORE = asyncio.Semaphore(1)`. Per-model semaphores allowed concurrent Ollama calls, overloading Mac Mini RAM and causing a timeout cascade (concurrent model loading → swapping → timeouts → agent restarts → more calls). A single global semaphore ensures only one model is loaded and inferring at any time.
- **Removed model aliases**: `model: fast` / `model: smart` aliases removed from `main.py`. All agents now specify the model name directly.
- **Unified model to `deepseek-r1:8b`**: all agents (`lighting`, `security`, `heating`, `cooling`, `voice`) and all config defaults (`smart_model`, `fast_model`, `orchestrator_model`) set to `deepseek-r1:8b`. Removes the need to pull multiple models.
- **Restored 300s intervals**: lighting and security agents restored to `decision_interval: 300`. Short intervals (30s, 60s) were tuned for reactive agents and caused unnecessary load without event-driven wake.
- **Tests** (22 new): global semaphore singleton, no per-model dict, concurrent different-model agents still serialise, VisionAgent both paths use no-arg `_get_llm_semaphore()`, no event-driven attributes on BaseAgent, UniversalAgent rejects `event_driven` kwarg, plain polling loop, deepseek defaults, agents.yaml and config.json assertions (122 total).
<br>
<br>

## [1.2.0] - 2026-04-12
### Added
- **Event-driven agent wake**: agents with `event_driven: true` in `agents.yaml` subscribe to HA `state_changed` events for their configured entities and wake *immediately* when any entity changes — no longer waiting out the full polling interval. A 10s debounce prevents rapid re-triggering. The polling interval becomes a heartbeat fallback.
- **Per-model LLM semaphores**: replaced the single global semaphore with a per-model-name dict. A slow `mistral:7b-instruct` call no longer blocks a fast `gemma4:e4b` agent. Each model serialises its own requests; different models run concurrently.
- **`model: fast` / `model: smart` aliases** in `agents.yaml`: resolve to the `fast_model` / `smart_model` add-on config values at runtime — no need to hardcode model tags per-agent.
- **`gemma4:e4b` as default `fast_model`**: lighting and security agents now default to the fast model for ~2s decisions vs ~20s with mistral:7b.
- **Lighting and security updated**: `event_driven: true`, `model: fast`, lighting heartbeat 60s, security heartbeat 30s.
- **Tests** (28 new): per-model semaphore isolation, no cross-model blocking, semaphore key correctness in `_call_llm` and VisionAgent paths, debounce conditions, subscription setup, model alias resolution, UniversalAgent propagation (120 total).
<br>
<br>

## [1.1.1] - 2026-04-12
### Fixed
- **LAN timeout for remote Ollama**: `ollama.Client` in both `base_agent.py` and `vision_agent.py` now constructed with `httpx.Timeout(connect=10, read=120, write=10, pool=10)`. The default 5s httpx read timeout was firing before responses could arrive from Ollama running on a separate machine (e.g. M4 Mac Mini) over a LAN connection, causing "LLM returned empty response" errors on every decision cycle.
- **M4-tuned generation params**: `num_ctx=4096` and `num_predict=1000` — restored to full values appropriate for a capable Ollama host, not capped for low-memory hardware.
- **Retry delays**: reduced from 5s/10s to 3s/8s — sized for transient LAN blips, not slow local hardware.
- **Tests** (12 new): httpx.Timeout present and read=120 on both clients, num_ctx/num_predict values, retry delays [3, 8], exhausted-retry ERROR string, VisionAgent Ollama text path params (92 total).
<br>
<br>

## [1.1.0] - 2026-04-12
### Added
- **Decision Export**: `GET /api/decisions/export` downloads rated decisions as JSONL (OpenAI fine-tuning format), JSON array, or CSV. Filter by `feedback=up|down` or all rated; scope by `agent_id`. Each JSONL row is a system/user/assistant message triple — ready to use directly with OpenAI fine-tuning or Axolotl/LLaMA-Factory.
- **Export Stats**: `GET /api/decisions/export/stats` returns counts of rated/unrated/up/down decisions — shown in the History tab before downloading so you know how many training examples are available.
- **Export UI**: History tab now shows a collapsible export panel with include-filter buttons (All rated / Thumbs up / Thumbs down), format selector (JSONL / JSON / CSV), live stat badges, and a Download button that triggers a browser file save.
- **Tests** (31 new): full coverage of JSONL row structure, filtering logic, stats counting, format serialization, and validation guards (80 total across all test suites).
<br>
<br>

## [1.0.9] - 2026-04-12
### Added
- **Entity Watchlist**: new sidebar tab ("Entity Watchlist") lets users pin any HA entity to monitor its live state. Cards group by domain with colour coding, show state prominently, expand to show attributes, and auto-refresh every 15s. Selections persist to localStorage.
- **Decision Feedback**: thumbs-up/down buttons on every History tab decision card. Rating is written back into the decision JSON file on disk with a `feedback_at` timestamp. Card backgrounds tint green/red to confirm the selection. Ratings persist across restarts.
- **`POST /api/decisions/feedback`**: saves thumbs-up/down to the correct decision file; validates `"up"` or `"down"` only; returns 404 if timestamp not found.
### Fixed
- **LLM serialization**: added a process-wide `asyncio.Semaphore(1)` in `base_agent.py` (`_get_llm_semaphore()`). All agents — including VisionAgent — now queue politely for Ollama. Previously all agents fired simultaneously at startup, overwhelming the model and causing cascading timeouts.
- **VisionAgent Ollama fallback**: when Gemini is not configured, `analyze_camera()` now calls `_analyze_with_ollama_text()` which fetches the camera entity state + nearby binary_sensor/sensor states and asks the configured Ollama model to reason about activity. Previously the agent entered an infinite sleep loop when Gemini was unavailable.
- **main.py**: passes `ollama_model` and `ollama_host` to VisionAgent constructor so the fallback model is configurable.
- **Tests** (14 new): semaphore singleton, serialisation, `_call_llm` acquires semaphore before `to_thread`, concurrent calls queue correctly, Vision fallback routing, `to_thread` usage, loop no longer deadlocks, feedback file I/O, `feedback_at` timezone-aware, value validation.
<br>
<br>

## [1.0.8] - 2026-04-12
### Added
- **Entity Manager UI**: "Entities" tab in the Agent Details panel lets users search all HA entities by ID or friendly name and add/remove them from any agent. Domain colour chips show entity type at a glance.
- **`GET /api/entities`**: returns all HA entities with `entity_id`, `friendly_name`, `state`, `domain`. Accepts optional `?domain=` filter.
- **`PATCH /api/factory/agents/{id}`**: accepts `{entities: [...]}` to update agent entity list; persists to `agents.yaml` and hot-reloads the in-memory agent immediately.
- **Tests** (16 new): entity list filtering, domain filter, sorted output, friendly_name fallback, YAML entity persistence, unrelated fields untouched, hot-reload, edge cases (empty list, missing agent, missing file).
<br>
<br>

## [1.0.7] - 2026-04-11
### Fixed
- **Timezone detection**: `main.py` queries `http://supervisor/core/api/config` at startup to detect the HA instance timezone, then calls `os.environ["TZ"] = tz; time.tzset()` to apply it to the running Python process. All `datetime.now()` calls (used for decision criteria like "is it evening?") now return correct local time.
- **Timezone-aware timestamps**: all `datetime.now().isoformat()` calls across agents and analytics replaced with `datetime.now().astimezone().isoformat()`, producing strings like `2026-04-11T20:30:00+05:30`. JavaScript `new Date(ts).toLocaleTimeString()` then converts correctly in the browser.
- **Analytics `_parse_ts()`**: new helper normalizes both naive and timezone-aware ISO strings to naive-local before comparison, eliminating `TypeError: can't compare offset-naive and offset-aware datetimes`.
- **Tests** (7 new): aware timestamp format, `_parse_ts` naive/UTC/offset/invalid, mixed-type comparison safety, broadcast timestamp includes offset.
<br>
<br>

## [1.0.6] - 2026-04-10
### Fixed
- **`asyncio.get_event_loop()` → `get_running_loop()`**: all 5 call sites in `ha_client.py` updated. Eliminates `DeprecationWarning` on Python 3.10+ and `RuntimeError` on 3.12+.
- **Callback exception isolation**: `_receive_messages()` wraps each subscription callback in `try/except`. A misbehaving callback no longer crashes the receiver loop and breaks all subsequent HA state updates.
- **Future done-guard**: `future.set_result()` is now guarded with `if not future.done()`. Prevents `asyncio.InvalidStateError` when HA sends a late response after a request timed out.
- **Non-blocking LLM calls**: `ollama.Client.chat()` and `generate_content()` calls in `base_agent.py`, `vision_agent.py`, `mcp_server.py`, and `orchestrator.py` moved to `asyncio.to_thread()`. The event loop is no longer blocked during LLM inference.
- **Tests** (12 new): `wait_until_connected`, `get_running_loop` usage, callback isolation, good callback still fires after bad one, late-response guard, cancelled-future guard, `to_thread` usage in `_call_llm`, non-blocking proof, orchestrator `plan()` + dashboard Ollama/Gemini paths.
<br>
<br>

## [1.0.0] - 2026-04-09
### Added
- **Voice integration**: `speak_tts` MCP tool calls `tts.speak` via `tts.google_ai_tts_2`; agents can now announce decisions aloud. `POST /api/voice/speak` endpoint for direct API calls.
- **Vision agent**: `VisionAgent` class monitors cameras via Gemini Vision API (`gemini-robotics-er-1.5-preview`). Doorbell and motion alerts spoken via TTS. Configurable per-camera in `agents.yaml`.
- **`get_camera_snapshot()`**: HA REST API camera proxy method added to `ha_client.py`.
- **Startup health gate**: `wait_until_connected(timeout=30)` before starting agent loops; `/api/health/ready` returns 503 while HA is still connecting.
- **Resilient `run.sh`**: removed `set -e`; Ollama pull failures no longer crash the add-on.
- **Exponential backoff** in reconnect loop: 10s → 20s → 40s → cap 60s.
### Fixed
- Guarded `google.generativeai` import in `orchestrator.py` and `vision_agent.py`.
- Unified all agents to `mistral:7b-instruct` in `agents.yaml` for maximum compatibility.
<br>
<br>

## [0.9.45] - 2025-12-22
### Fixed
- **HA Connectivity Robustness**: Improved error logging in `ha_client.py` with full URI and exception details. Added a startup wait period in `main.py` to prevent race conditions during early ingestion.
- **Custom HA URL Support**: Modified `run.sh` to allow user-defined `HA_URL` (from `options.json`) to take precedence even when a Long-Lived Access Token is provided.
- **Improved Reliability**: Architect suggestions and Knowledge Base ingestion now handle connection delays more gracefully.
<br>
<br>

## [0.9.44] - 2025-12-22
### Fixed
- **Chat Tool Execution**: Resolved `UnboundLocalError` (cannot access local variable 'params') when the AI Assistant triggers tools.
- **Agent Persistence**: Standardized `agents.yaml` pathing to ensure agents created through the UI are saved to the persistent `/config/agents.yaml` in Home Assistant Add-on environments.
<br>
<br>

## [0.9.43] - 2025-12-22
### Added
- **Gemini LLM Integration**: Added world-class LLM support for visual dashboard generation using Google Gemini.
- **Model Choice**: Users can now choose between local Ollama and Gemini (highly recommended for high-fidelity designs).
- **Robotics Preview Model**: Specifically added support for `gemini-robotics-er-1.5-preview` for advanced spatial and thermal visualizations.
- **Integration Settings**: New configuration fields in the UI for Gemini API Key, Model Selection, and a prioritization toggle.
- **Runtime Updates**: Gemini settings can be updated in-memory from the UI without requiring a full server restart.
<br>

## [0.9.42] - 2025-12-22
### Added
- **AI Visual Dashboard (Dynamic)**: Fully integrated natural language dashboard generation. Users can now command the dashboard style and focus via chat or a new dedicated UI tab.
- **Dynamic AI Prompting**: The Orchestrator now uses specific user instructions (e.g., "cyberpunk style", "security-focused") to architect the dashboard's HTML/CSS.
- **Background Refresh**: Implemented a periodic background loop that refreshes dashboard data every 5 minutes while preserving the user's requested aesthetic.
- **Direct UI Integration**: Dashboard is now a first-class citizen of the main UI, rendered via iframe with dedicated refresh controls.
### Fixed
- **Windows Pathing**: Resolved path normalization issues for `dynamic.html` on Windows, ensuring reliable dashboard file retrieval outside the Add-on environment.
- **Connectivity Guards**: Added safeguards to ensure Home Assistant is connected before attempting dashboard generation, preventing empty "no results" views.

## [0.9.41] - 2025-12-21
### Fixed
- **Docker Image Integrity**: Updated `Dockerfile` to correctly include `agents.yaml`, `skills/`, and `translations/` in the build, resolving issues with missing agents and tools in the Add-on environment.

## [0.9.40] - 2025-12-21
### Fixed
- **Connectivity Fallback**: Implemented automatic fallback to Direct Core Access (`http://homeassistant:8123`) when the Home Assistant Supervisor Token is missing in Add-on mode.
- **Agent Configuration Path**: Updated `agents.yaml` loading to prioritize `/config/agents.yaml` for persistent storage in Home Assistant Add-ons.

## [0.9.39] - 2025-12-21
### Fixed
- **Deployment**: Version bump to force update detection in Home Assistant (functional equivalent to v0.9.38).

## [0.9.38] - 2025-12-21
### Fixed
- **Instantiation Fix**: Corrected keyword arguments for global `HAWebSocketClient` instantiation in `main.py` (`url` -> `ha_url`).

## [0.9.37] - 2025-12-21
### Fixed
- **Connectivity Restoration**: Restored missing `ha_client` instantiation in `main.py` which was causing global `NoneType` errors.
- **Architect Stability**: Added runtime guards to `ArchitectAgent` to prevent crashes when Home Assistant is unreachable.

## [0.9.36] - 2025-12-21
### Fixed
- **Agent Initialization**: Fixed `AttributeError` in `BaseAgent` and `ArchitectAgent` by correctly positioning property definitions, ensuring `decision_dir` and `logger` are properly initialized.

## [0.9.35] - 2025-12-21
### Fixed
- **Orchestrator Init**: Fixed `AttributeError` by correctly positioning the `ha_client` property definition outside the `__init__` method, ensuring full initialization of all attributes.

## [0.9.34] - 2025-12-21
### Fixed
- **Startup Logic**: Fixed `Orchestrator` initialization scope issues and added robust `None` checks in `KnowledgeBase` to handle lazy connection availability.

## [0.9.33] - 2025-12-21
### Fixed
- **Deployment Verification**: Added `VERSION` tag to `MCPServer` and removed ambiguous docstring to verify fresh code deployment.

## [0.9.32] - 2025-12-21
### Fixed
- **Integrity Fix**: Restored `mcp_server.py` to a clean state to eliminate persistent syntax errors.

## [0.9.31] - 2025-12-21
### Fixed
- **Hotfix**: Resolved SyntaxError in `MCPServer` caused by artifacts in v0.9.30 release.

## [0.9.30] - 2025-12-21
### Fixed
- **Startup Crash Loop**: Refactored entire backend to use Lazy Injection for `ha_client`. Use `lambda: ha_client` to resolve the connection object at runtime, preventing components from holding a stale `None` reference.

## [0.9.29] - 2025-12-21
### Fixed
- **Knowledge Base Crash**: Guarded `ingest_ha_registry` against `NoneType` WebSocket error during startup loop.
- **Connection Logic**: Clarified need for `ha_access_token` when Supervisor API injection fails.

## [0.9.28] - 2025-12-21
### Fixed
- **Critical Crash Fix**: Resolved `NoneType` error in Universal Agent when Entity Discovery runs without a connection.
- **Port Consistency**: Verified and locked internal port to 8999 to resolve potential Ingress mismatches.

## [0.9.27] - 2025-12-21
### Fixed
- **Emergency Fix**: Guarded `ha_client` against `NoneType` crashes when disconnected.
- **Dashboard**: Relaxed Ingress path normalization to fix "black screen" 404 errors.
- **Diagnostics**: Added cleaner error handling for failed WebSocket message artifacts.

## [0.9.26] - 2025-12-21
### Added
- Specific static mount for `/assets` to ensure Ingress consistency.
- Environment diagnostics for Home Assistant connection tokens.
### Fixed
- Improved Supervisor detection in Add-on environment (checks for `/data/options.json`).
- Read `ha_access_token` from add-on options if environment variables are missing.
- Refined Ingress middleware to prevent double-slash asset 404s.

## [v0.9.25] - 2025-12-21
- **Stability Fix**: Hardened Home Assistant WebSocket client against `NoneType` crashes.
- **Connection Logic**: Improved Supervisor URL detection in Add-on environment.
- **Resilience**: Added connection guards to Knowledge Base ingestion and Universal Agents to prevent startup race conditions.

## [v0.9.24] - 2025-12-21
- **Ingress Fix**: Added `IngressMiddleware` path normalization to fix "Black Screen" asset issues.
- **Trailing Slashes**: Enforced trailing slashes for dashboard root to ensure relative asset loading.

## [v0.9.23] - 2025-12-21
- **Telemetry Silence**: Monkey-patched PostHog in memory to stop log spam from ChromaDB.
- **Diagnostics**: Added deep network diagnostics (DNS, Socket, HTTP) to verify Ollama connectivity.
 black-screen issues in Ingress.
- **Ingress Performance**: Smoothed out double-slash normalization for all backend routes.

## [0.9.22] - 2025-12-21

## [0.9.21] - 2025-12-20
