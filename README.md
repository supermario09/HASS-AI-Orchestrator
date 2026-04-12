# 🏠 HASS-AI-Orchestrator

![Version](https://img.shields.io/badge/version-v1.1.0-blue) ![Home Assistant](https://img.shields.io/badge/Home%20Assistant-Add--on-blue) ![Status](https://img.shields.io/badge/Status-Stable-green) ![Tests](https://img.shields.io/badge/tests-80%20passing-brightgreen)

**The Autonomous Multi-Agent Brain for your Smart Home.**

> This is a fork of [ITSpecialist111/HASS-AI-Orchestrator](https://github.com/ITSpecialist111/HASS-AI-Orchestrator), stabilized and extended with Voice, Vision, Entity Management, and AI Fine-Tuning capabilities.

The AI Orchestrator transforms your Home Assistant from a collection of manual toggles and rigid automations into a dynamic, thinking ecosystem. It deploys **Autonomous AI Agents** that reason about your home's state, understand your intent, and execute actions intelligently.

---

## ✨ What's New (v1.0.x → v1.1.0)

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

---

## ⚡ Performance Notes

- Agents stagger at startup (30s apart) to avoid LLM pile-ons.
- A global LLM semaphore ensures only one Ollama call runs at a time — preventing timeouts.
- Default decision interval is 120s. Set to 5–10s for "real-time" mode (GPU recommended).

---

## 🧪 Test Suite

80 automated tests covering all stability fixes, entity management, vision fallback, feedback, and export logic.

```bash
cd ai-orchestrator/backend
pytest tests/test_stability_fixes.py tests/test_v1_0_8_entity_manager.py \
       tests/test_v1_0_9_fixes.py tests/test_v1_1_0_export.py -v
```

---

## ⚠️ Requirements

- **Home Assistant OS** or Supervised
- **Ollama Server** (local or network-accessible)
- **Hardware**: Raspberry Pi 4 (8GB) or NUC recommended. Dedicated GPU server recommended for real-time agent speed.
- **Optional**: Google Gemini API key for Vision analysis and AI Visual Dashboard

---

[Read Full Documentation](ai-orchestrator/README.md)
