import asyncio
import sys
import json
from pathlib import Path
import httpx

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

try:
    from telco_network_optimizer_agent.mg_prompt import MERGED_MCP_PROMPT
except ImportError:
    from mg_prompt import MERGED_MCP_PROMPT


# Windows subprocess fix
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# Load MCP config
CONFIG_PATH = Path(__file__).resolve().parent / "mcp_servers.json"
with open(CONFIG_PATH, "r") as f:
    mcp_config = json.load(f)["mcpServers"]

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


# -----------------------------
#  OPENROUTER CLIENT
# -----------------------------
class OpenRouterAgent:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = system_prompt
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("⚠ OPENROUTER_API_KEY not set in environment variables")

    async def run(self, user_query: str):
        async with httpx.AsyncClient(timeout=120) as client:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query}
                ]
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Telco Ops AI Agent"
            }

            response = await client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


# Build root agent with Grok 4.1 Free
root_agent = OpenRouterAgent(
    model="grok-2-latest",   # Grok 4.1 Free model alias
    system_prompt=MERGED_MCP_PROMPT
)
