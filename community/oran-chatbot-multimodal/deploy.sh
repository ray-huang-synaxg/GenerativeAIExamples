#!/usr/bin/env bash
# deploy.sh — One-shot setup and launch for oran-chatbot-multimodal
# Usage:
#   ./deploy.sh              — install deps and start server on port 8011
#   ./deploy.sh --port 8080  — use a custom port
#   ./deploy.sh --help       — show this help
#
# Requirements: Python 3.10–3.12, git, uv (or pip)

set -euo pipefail

PORT=8011
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,8p' "$0" | sed 's/^# //'
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════"
echo "  ORAN Chatbot — Deploy Script"
echo "  Directory : $SCRIPT_DIR"
echo "  Port      : $PORT"
echo "═══════════════════════════════════════════════════"

# ── 1. Python check ───────────────────────────────────────────────────────────
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || true)
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Python 3.10+ not found. Install it and retry."
  exit 1
fi
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/5] Python: $PYTHON ($PY_VER)"

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
  echo "[2/5] Creating virtual environment..."
  if command -v uv &>/dev/null; then
    uv venv .venv --python "$PYTHON"
  else
    "$PYTHON" -m venv .venv
  fi
else
  echo "[2/5] Virtual environment already exists — skipping creation."
fi

VENV_PYTHON=".venv/bin/python"

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo "[3/5] Installing dependencies (this may take a few minutes)..."
if command -v uv &>/dev/null; then
  uv pip install --python "$VENV_PYTHON" -r requirements.txt
else
  "$VENV_PYTHON" -m pip install --upgrade pip -q
  "$VENV_PYTHON" -m pip install -r requirements.txt -q
fi
echo "      Dependencies installed."

# ── 4. Config check ───────────────────────────────────────────────────────────
echo "[4/5] Checking config.yaml..."
if [[ ! -f "config.yaml" ]]; then
  echo "ERROR: config.yaml not found."
  exit 1
fi

OPENAI_FLAG=$("$VENV_PYTHON" -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('OPENAI', False))")
if [[ "$OPENAI_FLAG" == "True" ]]; then
  OPENAI_KEY=$("$VENV_PYTHON" -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('openai_api_key',''))")
  OPENAI_URL=$("$VENV_PYTHON" -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('openai_base_url',''))")
  OPENAI_MODEL=$("$VENV_PYTHON" -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('openai_model_name',''))")
  echo "      LLM backend : OpenAI-compatible"
  echo "      Model       : $OPENAI_MODEL"
  echo "      Base URL    : ${OPENAI_URL:-api.openai.com}"
  if [[ "$OPENAI_KEY" == "sk-placeholder" || -z "$OPENAI_KEY" ]]; then
    echo ""
    echo "  ⚠️  WARNING: openai_api_key is still a placeholder in config.yaml."
    echo "     Edit config.yaml and set your real API key before prompting."
    echo ""
  fi
else
  NVIDIA_KEY=$("$VENV_PYTHON" -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('nvidia_api_key',''))")
  echo "      LLM backend : NVIDIA API Catalog"
  if [[ "$NVIDIA_KEY" == "nvapi--***" || -z "$NVIDIA_KEY" ]]; then
    echo ""
    echo "  ⚠️  WARNING: nvidia_api_key is still a placeholder in config.yaml."
    echo "     Edit config.yaml and set your real NVIDIA API key before prompting."
    echo ""
  fi
fi

# ── 5. Launch ─────────────────────────────────────────────────────────────────
echo "[5/5] Starting Streamlit on port $PORT..."
echo ""
echo "  → Open http://localhost:$PORT in your browser"
echo "  → Press Ctrl+C to stop"
echo ""

exec "$VENV_PYTHON" -m streamlit run Multimodal_Assistant.py \
  --server.port "$PORT" \
  --server.headless true \
  --server.address 0.0.0.0
