# 🏠 HASS-AI-Orchestrator

![Version](https://img.shields.io/badge/version-v1.2.3-blue) ![Home Assistant](https://img.shields.io/badge/Home%20Assistant-Add--on-blue) ![Status](https://img.shields.io/badge/Status-Stable-green) ![Tests](https://img.shields.io/badge/tests-134%20passing-brightgreen)

**The Autonomous Multi-Agent Brain for your Smart Home.**

> This is a fork of [ITSpecialist111/HASS-AI-Orchestrator](https://github.com/ITSpecialist111/HASS-AI-Orchestrator), stabilized and extended with Voice, Vision, Entity Management, and AI Fine-Tuning capabilities.

The AI Orchestrator transforms your Home Assistant from a collection of manual toggles and rigid automations into a dynamic, thinking ecosystem. It deploys **Autonomous AI Agents** that reason about your home's state, understand your intent, and execute actions intelligently.

---

## ✨ What's New (v1.0.x → v1.2.3)

### v1.2.3 — Fix Empty LLM Responses
- **Restored `num_predict=1000`**: deepseek-r1:8b generates a `<think>` block before every response even when `think: False` is set (silently ignored on some Ollama versions). With the v1.2.2 cap of 500 tokens, the think block consumed the entire budget — after stripping it, content was empty and every decision failed with "LLM returned empty response". 1000 tokens gives room for a think block plus the full JSON response.
- **Fixed warmup prompt**: the `"ping"` warmup with `max_tokens=3` hit the same issue. Now uses an explicit prompt with `max_tokens=200` so the warmup reliably succeeds.

### v1.2.2 — LLM Efficiency
- **`keep_alive=-1`**: model stays loaded in Ollama memory permanently — eliminates the 5-15s model-reload cost that was hitting every other decision cycle at 300s intervals
- **`num_ctx` halved to 2048**: decision prompts fit in 600-1200 tokens; cutting the KV cache halves prefill time and reduces VRAM pressure
- **Temperature 0.7 → 0.3**: more deterministic JSON, slightly faster sampling
- **Model warmup before first decision**: agents ping the model on startup so the first real decision doesn't pay the load penalty
- **Services cache in agents**: `get_services()` now cached 10 minutes instead of fetching on every 300s cycle

### v1.2.1 — Stability Revert
- **Reverted event-driven wake**: removed reactive subscriptions — they increased Ollama call frequency and worsened timeouts. Use HA native automations for reactive triggers; the AI layer handles slow decisioning only.
- **Single global semaphore restored**: per-model semaphores allowed concurrent Ollama calls, overloading RAM on the Mac Mini. One global `asyncio.Semaphore(1)` ensures only one model loads at a time — the correct design for a single Ollama server.
- **Unified model to `deepseek-r1:8b`**: one model for all agents — no need to pull `gemma4:e4b` or `mistral:7b-instruct`. Set in `smart_model`, `fast_model`, and `orchestrator_model`.
- **Restored 300s polling intervals**: lighting and security agents back to 5-minute cycles matching the AI decisioning cadence.

### v1.2.0 — Fast Reactive Agents (reverted in v1.2.1)

### v1.1.1 — LAN Ollama Timeout Fix
- **Eliminates "LLM returned empty response"** when Ollama runs on a separate machine (e.g. M4 Mac Mini) over a LAN connection — root cause was httpx's default 5s read timeout firing before the response could arrive
- Explicit `httpx.Timeout(connect=10, read=120)` applied to all Ollama clients in `base_agent.py` and `vision_agent.py`
- Generation params tuned for capable hardware: `num_ctx=4096`, `num_predict=1000`
- Retry delays tightened to 3s/8s (LAN blips, not slow local hardware)

### v1.1.0 — Decision Export for Fine-Tuning
- **Export rated decisions** as JSONL (OpenAI fine-tuning format), JSON, or CSV
- Filter by thumbs-up only, thumbs-down only, or all rated decisions
- Scope export per-agent or across all agents
- Live stats panel shows how many decisions are rated before downloading
- `GET /api/decisions/export` and `GET /api/decisions/export/stats` endpoints

### v1.0.9 — LLM Serialization + Vision Fallback + Decision Feedback + Entity Watchlist
- **Global LLM semaphore**: agents now queue politely for Ollama — no more thundering-herd timeouts when all agents fire simultaneously
- **Vision fallback**: VisionAgent uses Gemini Vision when configured, falls back to Ollama text-based analysis of camera entity state + nearby sensors when Gemini is unavailable
- **Decision feedback**: thumbs-up/down buttons on every decision card in the History tab; ratings persisted to the decision JSON files for future fine-tuning
- **Entity Watchlist**: new sidebar tab to pin any HA entity, watch live state grouped by domain, auto-refreshes every 15s, persists in browser localStorage

### v1.0.8 — Entity Manager UI
- Edit the entities any agent monitors directly from the dashboard — no more editing `agents.yaml` by hand
- Search all 2000+ HA entities by ID or friendly name with live autocomplete
- Changes persist to `agents.yaml` and hot-reload the in-memory agent immediately

### v1.0.7 — Timezone Fix
- Auto-detects Home Assistant timezone via the Supervisor API at startup
- All decision timestamps, logs, and UI times now display in the correct local timezone
- Fixed `TypeError` when comparing naive vs timezone-aware timestamps in analytics

### v1.0.6 — Core Stability
- Fixed WebSocket `get_event_loop()` deprecation warning (Python 3.10+)
- Fixed callback exception isolation — one bad subscription no longer kills the receiver loop
- Fixed race condition: `future.set_result()` on an already-resolved future no longer crashes
- All LLM calls (`ollama.Client.chat`, `generate_content`) moved to `asyncio.to_thread()` to unblock the event loop

### v1.0.0 — Voice + Vision (Original Stable Release)
- **Voice**: `speak_tts` MCP tool calls HA TTS (`tts.speak` via `tts.google_ai_tts_2`)
- **Vision**: VisionAgent monitors cameras using Gemini Vision; doorbell alerts spoken via TTS
- **Resilient startup**: removed `set -e` from `run.sh`, add-on no longer crashes on Ollama timeouts

---

## 🌟 Core Features

### 🧠 Real Reasoning, Not Scripts
Instead of writing complex YAML, tell an agent in plain English:
> *"You are the Security Guard. If the front porch is occupied for more than 5 minutes at night, turn on the floodlight and notify me."*

### 🗣️ Natural Language Chat
A **Floating Chat Assistant** lives in your dashboard. Talk to your house naturally:
> *"It's movie night. Get the living room ready."*
> *"Who left the garage door open?"*

### 🔧 No-Code Agent Factory
The **Architect AI** will interview you and build the perfect agent for your needs automatically. It even surfaces **Smart Suggestions** based on your devices.

### 👁️ Entity Watchlist
Pin any Home Assistant entity and watch its live state from a dedicated sidebar panel. Groups by domain, shows attributes on demand, auto-refreshes every 15 seconds.

### 📊 Entity Manager
Assign specific entities to any agent from the dashboard — no YAML editing required. Changes persist across restarts.

### 🎤 Voice Announcements
Agents can speak decisions aloud via Google AI TTS. High-impact actions trigger automatic announcements before executing.

### 📷 Camera Vision
The VisionAgent monitors your cameras on a configurable interval. Uses Gemini Vision when a key is configured; falls back to Ollama text-based analysis of entity state + motion sensors when Gemini is unavailable.

### 🎨 AI Visual Dashboard
Experience high-fidelity visualizations generated in real-time. Command the dashboard style via natural language — no YAML, no card configuration.

### 🏋️ Fine-Tuning Export
Rate agent decisions with thumbs-up/down. Export all rated decisions as JSONL in OpenAI fine-tuning format to improve model quality over time.

---

## 📦 Installation

### Option A — Install from this fork

1. **Add Repository**:
   Copy this URL:
   ```
   https://github.com/supermario09/HASS-AI-Orchestrator
   ```
   Go to **Home Assistant > Settings > Add-ons > Add-on Store > ⋮ > Repositories** and add it.

2. **Install "AI Orchestrator"** from the list.

3. **Configure** in the Add-on Configuration tab:

   | Option | Description |
   |--------|-------------|
   | `ollama_host` | Your Ollama server URL (e.g. `http://192.168.1.x:11434`) |
   | `ha_access_token` | Long-Lived Access Token from your HA profile |
   | `smart_model` | Ollama model for agents (e.g. `mistral:7b-instruct`) |
   | `gemini_api_key` | (Optional) Google Gemini API key for Vision + Dashboard |
   | `use_gemini_for_vision` | Enable camera analysis via Gemini Vision |
   | `use_gemini_for_dashboard` | Enable AI visual dashboard via Gemini |
   | `dry_run_mode` | `true` = agents plan but don't execute (safe default) |
   | `decision_interval` | Seconds between agent decision cycles (default: 120) |

4. **Pull Ollama Models**:
   ```bash
   ollama pull mistral:7b-instruct
   # Optional — only needed if ENABLE_RAG=true:
   ollama pull nomic-embed-text
   ```

5. **Start & Explore**: open the Web UI to see your agents thinking in real-time.

---

## 🔒 Privacy & Safety

- **100% Local**: works with your local Ollama instance — no data leaves your network unless you configure Gemini.
- **Domain Allowlist**: AI only controls safe domains (`light`, `switch`, `climate`, etc.) by default.
- **Blocked Domains**: dangerous domains (`shell_command`, `hassio`, `script`, `automation`) are explicitly blocked.
- **Approval Queue**: high-impact actions (unlocking doors, disarming alarms) require manual approval from the dashboard.
- **Dry Run Mode**: agents plan but don't execute until you trust them.
- **Event-driven, not polling**: reactive agents subscribe to HA state changes and wake in <1s rather than waiting a polling interval.

---

## ⚡ Performance Notes

- Agents stagger at startup (30s apart) to avoid LLM pile-ons.
- A global LLM semaphore ensures only one Ollama call runs at a time — preventing timeouts.
- Default decision interval is 120s. Set to 5–10s for "real-time" mode (GPU recommended).

---

## 🧪 Test Suite

134 automated tests covering all stability fixes, entity management, vision fallback, feedback, export, LAN timeout, revert correctness, and LLM efficiency parameters.

```bash
cd ai-orchestrator/backend
pytest tests/test_stability_fixes.py tests/test_v1_0_8_entity_manager.py \
       tests/test_v1_0_9_fixes.py tests/test_v1_1_0_export.py \
       tests/test_v1_1_1_lan_timeout.py tests/test_v1_2_0_fast_reactive.py -v
```

---

## ⚠️ Requirements

- **Home Assistant OS** or Supervised
- **Ollama Server** (local or network-accessible)
- **Hardware**: Raspberry Pi 4 (8GB) or NUC recommended. Dedicated GPU server recommended for real-time agent speed.
- **Optional**: Google Gemini API key for Vision analysis and AI Visual Dashboard

---

[Read Full Documentation](ai-orchestrator/README.md)
