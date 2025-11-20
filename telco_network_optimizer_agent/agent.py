import asyncio
import sys
import json
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
try:
    from telco_network_optimizer_agent.mg_prompt import MERGED_MCP_PROMPT
except ImportError:
    from mg_prompt import MERGED_MCP_PROMPT

# Fix for Windows subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load JSON config
CONFIG_PATH = Path(__file__).resolve().parent / "mcp_servers.json"
with open(CONFIG_PATH, "r") as f:
    mcp_config = json.load(f)["mcpServers"]

# # Create a list of MCPToolset instances for each server
# toolsets = []
# for name, config in mcp_config.items():
#     print(f"Adding MCP Server: {name}")
#     toolsets.append(
#         MCPToolset(
#             connection_params=StdioServerParameters(
#                 command=config["command"],
#                 args=config["args"]
#             )
#         )
#     )

# Create a list of MCPToolset instances for each server
toolsets = []
for name, config in mcp_config.items():
    print(f"Adding MCP Server: {name}")
    try:
        toolset = MCPToolset(
            connection_params=StdioServerParameters(
                command=config["command"],
                args=config["args"]
            )
        )
        print(f"→ Toolset created for {name}: {toolset}")
        toolsets.append(toolset)
    except Exception as e:
        print(f"✗ Failed to create toolset for {name}: {e}")
        continue

    
# Build the root agent with all MCP servers
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="tel_ops_ai_agent",
    instruction=MERGED_MCP_PROMPT,
    tools=toolsets  # Multiple MCP servers
)


