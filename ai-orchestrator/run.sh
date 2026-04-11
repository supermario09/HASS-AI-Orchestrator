#!/usr/bin/env bash
# DO NOT use set -e — it causes the add-on to exit (and HA to restart it) on any
# non-zero command, including transient model-pull failures or missing config keys.

echo "=========================================="
echo "AI Orchestrator - Starting up"
echo "=========================================="

# Parse add-on configuration
CONFIG_PATH="/data/options.json"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "WARNING: Configuration file not found at $CONFIG_PATH — using environment variable defaults."
    # Continue with defaults rather than exiting
fi

# Helper: read a JSON key with fallback default
jq_or_default() {
    local key="$1"
    local default="$2"
    if [ -f "$CONFIG_PATH" ]; then
        local val
        val=$(jq -r "${key} // \"${default}\"" "$CONFIG_PATH" 2>/dev/null)
        echo "${val:-$default}"
    else
        echo "$default"
    fi
}

# Extract configuration values
export OLLAMA_HOST=$(jq_or_default '.ollama_host' "http://localhost:11434")
export DRY_RUN_MODE=$(jq_or_default '.dry_run_mode' "true")
export LOG_LEVEL=$(jq_or_default '.log_level' "info" | tr '[:lower:]' '[:upper:]')
export ORCHESTRATOR_MODEL=$(jq_or_default '.orchestrator_model' "mistral:7b-instruct")
export SMART_MODEL=$(jq_or_default '.smart_model' "mistral:7b-instruct")
export FAST_MODEL=$(jq_or_default '.fast_model' "mistral:7b-instruct")
export DECISION_INTERVAL=$(jq_or_default '.decision_interval' "120")
export ENABLE_GPU=$(jq_or_default '.enable_gpu' "false")

# Security Controls
export ALLOWED_DOMAINS=$(jq_or_default '.allowed_domains' "")
export BLOCKED_DOMAINS=$(jq_or_default '.blocked_domains' "")
export HIGH_IMPACT_SERVICES=$(jq_or_default '.high_impact_services' "")
export MIN_TEMP=$(jq_or_default '.min_temp' "10.0")
export MAX_TEMP=$(jq_or_default '.max_temp' "30.0")
export MAX_TEMP_CHANGE=$(jq_or_default '.max_temp_change' "3.0")
export GEMINI_API_KEY=$(jq_or_default '.gemini_api_key' "")
export USE_GEMINI_FOR_DASHBOARD=$(jq_or_default '.use_gemini_for_dashboard' "false")
export GEMINI_MODEL_NAME=$(jq_or_default '.gemini_model_name' "gemini-robotics-er-1.5-preview")

# Home Assistant API configuration
export SUPERVISOR_TOKEN="${SUPERVISOR_TOKEN}"
export HA_ACCESS_TOKEN=$(jq_or_default '.ha_access_token' "")
export HA_URL=$(jq_or_default '.ha_url' "http://supervisor/core")

echo "Configuration loaded:"
echo "  Ollama Host: $OLLAMA_HOST"
echo "  Dry Run Mode: $DRY_RUN_MODE"
echo "  Log Level: $LOG_LEVEL"
echo "  Orchestrator Model: $ORCHESTRATOR_MODEL"
echo "  Smart Model: $SMART_MODEL"
echo "  Fast Model: $FAST_MODEL"
echo "  Decision Interval: ${DECISION_INTERVAL}s"
echo "  GPU Enabled: $ENABLE_GPU"
if [ -n "$HA_ACCESS_TOKEN" ]; then
    echo "  HA Access Token: PROVIDED (Length: ${#HA_ACCESS_TOKEN})"
    # Switch to Direct Core Access to bypass Supervisor Proxy issues ONLY if still using default proxy URL
    if [ "$HA_URL" == "http://supervisor/core" ]; then
        export HA_URL="http://homeassistant:8123"
        echo "  > Switching to Direct Core Access: $HA_URL"
    else
        echo "  > Using custom HA URL: $HA_URL"
    fi
else
    echo "  HA Access Token: NOT PROVIDED (Using Supervisor Token fallback)"
fi

# Start Ollama server if using localhost
if [[ "$OLLAMA_HOST" == *"localhost"* ]] || [[ "$OLLAMA_HOST" == *"127.0.0.1"* ]]; then
    echo "=========================================="
    echo "Starting Ollama server..."
    echo "=========================================="

    # Set GPU support if enabled
    if [ "$ENABLE_GPU" = "true" ]; then
        export OLLAMA_GPU=1
        echo "GPU support enabled"
    fi

    # Start Ollama in background
    ollama serve &
    OLLAMA_PID=$!

    # Wait for Ollama to be ready (up to 60 seconds)
    echo "Waiting for Ollama to start..."
    OLLAMA_READY=0
    for i in $(seq 1 60); do
        if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo "Ollama is ready!"
            OLLAMA_READY=1
            break
        fi
        sleep 1
    done

    if [ "$OLLAMA_READY" -eq 0 ]; then
        echo "WARNING: Ollama did not start within 60 seconds. Backend will attempt to connect later."
        # Do NOT exit — the backend's reconnect logic will retry
    fi

    # Pull models if not already present — failures are non-fatal
    echo "=========================================="
    echo "Checking required models..."
    echo "=========================================="

    if [ "$OLLAMA_READY" -eq 1 ]; then
        if ! ollama list 2>/dev/null | grep -q "${SMART_MODEL%%:*}"; then
            echo "Smart Model not found. Pulling $SMART_MODEL (this may take a while)..."
            ollama pull "$SMART_MODEL" || echo "WARNING: Failed to pull $SMART_MODEL — will retry on next restart"
        else
            echo "Smart model $SMART_MODEL already available."
        fi

        if ! ollama list 2>/dev/null | grep -q "${FAST_MODEL%%:*}"; then
            echo "Fast Model not found. Pulling $FAST_MODEL..."
            ollama pull "$FAST_MODEL" || echo "WARNING: Failed to pull $FAST_MODEL — will retry on next restart"
        else
            echo "Fast model $FAST_MODEL already available."
        fi
    else
        echo "Skipping model checks — Ollama not reachable."
    fi
fi

# Create necessary directories
mkdir -p /data/decisions /data/logs /data/chroma /data/manuals /data/dashboard
echo "Data directories created."

# Phase 6: Ensure agents.yaml exists in persistent config
if [ ! -f /config/agents.yaml ]; then
    echo "Creating default agents.yaml in /config..."
    if [ -f /app/agents.yaml ]; then
        cp /app/agents.yaml /config/agents.yaml
    else
        echo "agents: []" > /config/agents.yaml
    fi
fi

# Link /config/agents.yaml to where the app expects it
rm -f /app/backend/agents.yaml
ln -sf /config/agents.yaml /app/backend/agents.yaml
echo "Linked agents.yaml to persistent storage"

echo "=========================================="
echo "Starting FastAPI Backend"
echo "=========================================="

# Start FastAPI backend — use exec so signals are forwarded correctly
cd /app/backend
exec python3 -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8999 \
    --log-level "${LOG_LEVEL,,}" \
    --no-access-log
