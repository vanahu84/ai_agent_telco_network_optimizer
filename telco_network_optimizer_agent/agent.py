import asyncio
import sys
import json
import os
from pathlib import Path

import openai
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

# Create MCP toolsets (unchanged)
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

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

# Configure OpenAI API key (expects OPENAI_API_KEY env var)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
openai.api_key = openai_api_key

# A small async wrapper class that behaves similarly to the original LlmAgent constructor usage.
# It sends the merged MCP prompt as the system instruction and exposes an async `chat` method.
class OpenAIAgent:
    def __init__(self, *, model: str, name: str, instruction: str, tools: list):
        self.model = model
        self.name = name
        self.instruction = instruction
        self.tools = tools

    async def chat(self, user_message: str, **kwargs) -> str:
        """Send a message to OpenAI ChatCompletion and return the assistant text.

        This uses the chat completions endpoint in async form. It includes the agent's
        `instruction` as the system message so the model knows how to behave with MCP tools.
        """
        # Build conversation
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": user_message},
        ]

        # Use the async create if available; otherwise fall back to sync call inside a thread
        try:
            # openai.ChatCompletion.acreate is available in some openai versions
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            content = response["choices"][0]["message"]["content"]
            return content
        except AttributeError:
            # Fallback: run blocking call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(model=self.model, messages=messages, **kwargs),
            )
            content = response["choices"][0]["message"]["content"]
            return content

    def __repr__(self):
        return f"<OpenAIAgent name={self.name} model={self.model} tools={len(self.tools)}>"


# Create the root agent using OpenAI model instead of Gemini
root_agent = OpenAIAgent(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # change model via OPENAI_MODEL env var
    name="tel_ops_ai_agent",
    instruction=MERGED_MCP_PROMPT,
    tools=toolsets,
)

# Example usage
async def main():
    # Example prompt (replace with real input flow in your application)
    resp = await root_agent.chat("Hello — list the available MCP tools and their names.")
    print("Agent response:\n", resp)

if __name__ == "__main__":
    asyncio.run(main())
