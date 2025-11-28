"""
Timeout-Resistant Agent for Telecom Network Optimization
Now using OpenAI/OpenRouter instead of Google Gemini
"""
import asyncio
import sys
import json
import os
from pathlib import Path

try:
    from telco_network_optimizer_agent.mg_prompt import MERGED_MCP_PROMPT
    from telco_network_optimizer_agent.agent_unified import create_agent
except ImportError:
    from mg_prompt import MERGED_MCP_PROMPT
    from agent_unified import create_agent

# Fix for Windows subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Determine which provider to use from environment
provider = None
if os.getenv("OPENAI_API_KEY"):
    provider = "openai"
    print("✓ Using OpenAI API")
elif os.getenv("OPENROUTER_API_KEY"):
    provider = "openrouter"
    print("✓ Using OpenRouter API")
else:
    print("⚠ Warning: No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY")

# Create timeout-resistant agent
try:
    root_agent = create_agent(provider=provider)
    print(f"✓ Timeout-resistant agent created: {root_agent}")
except Exception as e:
    print(f"✗ Failed to create agent: {e}")
    root_agent = None