#!/bin/bash
set -e

echo "=========================================="
echo "  Telecom Network Optimizer - Starting"
echo "=========================================="

# Check API keys
if [ -z "$OPENAI_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  WARNING: No API keys found!"
    echo "Set OPENAI_API_KEY or OPENROUTER_API_KEY"
fi

# Display config
echo ""
echo "Configuration:"
[ -n "$OPENAI_API_KEY" ] && echo "  ✓ OpenAI API Key: Set"
[ -n "$OPENROUTER_API_KEY" ] && echo "  ✓ OpenRouter API Key: Set"
echo "  Port: ${PORT:-7860}"
echo ""

# Python version
echo "🐍 Python version:"
python --version

# Verify core dependencies
echo ""
echo "📦 Checking dependencies..."
python -c "import fastapi; import uvicorn; import openai; import httpx; print('✓ Dependencies OK')" || {
    echo "❌ Missing dependencies!"
    echo "Installing..."
    pip install -r requirements.txt
}

# Start web server
echo ""
echo "=========================================="
echo "  Starting Web Server"
echo "=========================================="
echo "  Host: 0.0.0.0"
echo "  Port: ${PORT:-7860}"
echo "=========================================="
echo ""

exec python telco_network_optimizer_agent/web_server.py
