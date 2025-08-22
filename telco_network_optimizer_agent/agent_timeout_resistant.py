import asyncio
import sys
import json
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Fix for Windows subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load JSON config with error handling
CONFIG_PATH = Path(__file__).resolve().parent / "mcp_servers.json"
try:
    with open(CONFIG_PATH, "r") as f:
        mcp_config = json.load(f)["mcpServers"]
except Exception as e:
    print(f"Error loading MCP config: {e}")
    mcp_config = {}

# Create toolsets with timeout handling
toolsets = []
for name, config in mcp_config.items():
    if config.get("disabled", False):
        print(f"Skipping disabled server: {name}")
        continue
        
    print(f"Adding MCP Server: {name}")
    try:
        # Create toolset with timeout settings
        toolset = MCPToolset(
            connection_params=StdioServerParameters(
                command=config["command"],
                args=config["args"],
                env=config.get("env", {}),
                timeout=config.get("timeout", 10.0)
            )
        )
        toolsets.append(toolset)
        print(f"Successfully added {name}")
    except Exception as e:
        print(f"Failed to create toolset for {name}: {e}")
        continue

# Only create agent if we have working toolsets
if toolsets:
    try:
        try:
            from telco_network_optimizer_agent.mg_prompt import MERGED_MCP_PROMPT
        except ImportError:
            from mg_prompt import MERGED_MCP_PROMPT
        
        root_agent = LlmAgent(
            model="gemini-2.0-flash",
            name="timeout_resistant_mcp_agent",
            instruction=MERGED_MCP_PROMPT,
            tools=toolsets
        )
        print(f"Created agent with {len(toolsets)} working MCP servers")
    except Exception as e:
        print(f"Failed to create agent: {e}")
        root_agent = None
else:
    print("No working MCP servers found - agent not created")
    root_agent = None