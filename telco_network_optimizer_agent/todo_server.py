import asyncio
from typing import Any
import json
import logging
import os

from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type
import mcp.server.stdio
from google.adk.tools.function_tool import FunctionTool
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)

# TODO_FILE_PATH = os.path.join(os.path.dirname(__file__), "todo_list.txt")
TODO_FILE_PATH = "/tmp/todo_list.txt"


# --- File Utility Functions ---
def read_tasks() -> list:
    if not os.path.exists(TODO_FILE_PATH):
        return []
    with open(TODO_FILE_PATH, "r") as file:
        return [line.strip() for line in file.readlines()]

def write_tasks(tasks: list) -> None:
    with open(TODO_FILE_PATH, "w") as file:
        for task in tasks:
            file.write(f"{task}\n")

# --- Tool Functions with dict inputs ---
def add_task(input: dict) -> dict[str, Any]:
    task = input.get("task")
    if not task:
        return {"success": False, "message": "Missing 'task' in input."}
    tasks = read_tasks()
    tasks.append(task)
    write_tasks(tasks)
    return {"success": True, "message": "Task added successfully."}

def delete_task(input: dict) -> dict[str, Any]:
    task = input.get("task")
    if not task:
        return {"success": False, "message": "Missing 'task' in input."}
    tasks = read_tasks()
    if task in tasks:
        tasks.remove(task)
        write_tasks(tasks)
        return {"success": True, "message": "Task deleted successfully."}
    else:
        return {"success": False, "message": "Task not found."}

def list_tasks(input: dict) -> dict[str, Any]:
    tasks = read_tasks()
    return {"success": True, "tasks": tasks}

# --- MCP Server Setup ---
logging.info("Creating MCP Server instance for TODO list...")
app = Server("todo-list-mcp-server")

# Register tools — no schema issues since input is always a dict
ADK_TODO_TOOLS = {
    "add_task": FunctionTool(func=add_task),
    "list_tasks": FunctionTool(func=list_tasks),
    "delete_task": FunctionTool(func=delete_task),
}

@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    logging.info("MCP Server: Listing tools...")
    mcp_tools_list = []
    for tool_name, adk_tool_instance in ADK_TODO_TOOLS.items():
        try:
            if not adk_tool_instance.name:
                adk_tool_instance.name = tool_name

            mcp_tool_schema = adk_to_mcp_tool_type(adk_tool_instance)
            logging.info(f"Generated tool schema: {mcp_tool_schema}")
            if mcp_tool_schema is None:
                logging.error("Generated tool schema is None.")
                continue

            logging.info(f"MCP Server: Advertising tool: {mcp_tool_schema.name}, InputSchema: {mcp_tool_schema.inputSchema}")
            mcp_tools_list.append(mcp_tool_schema)

        except Exception as e:
            logging.error(f"Error processing tool {tool_name}: {e}", exc_info=True)
            continue

    return mcp_tools_list

@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    logging.info(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    if name in ADK_TODO_TOOLS:
        adk_tool_instance = ADK_TODO_TOOLS[name]
        try:
            adk_tool_response = await adk_tool_instance.run_async(
                args=arguments,
                tool_context=None,
            )
            logging.info(f"MCP Server: ADK tool '{name}' executed. Response: {adk_tool_response}")
            response_text = json.dumps(adk_tool_response, indent=2)
            return [mcp_types.TextContent(type="text", text=response_text)]

        except Exception as e:
            logging.error(f"MCP Server: Error executing ADK tool '{name}': {e}", exc_info=True)
            error_payload = {
                "success": False,
                "message": f"Failed to execute tool '{name}': {str(e)}",
            }
            return [mcp_types.TextContent(type="text", text=json.dumps(error_payload))]
    else:
        error_payload = {
            "success": False,
            "message": f"Tool '{name}' not implemented by this server.",
        }
        return [mcp_types.TextContent(type="text", text=json.dumps(error_payload))]

# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logging.info("MCP Stdio Server: Starting handshake with client...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name,
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        logging.info("MCP Stdio Server: Run loop finished or client disconnected.")

if __name__ == "__main__":
    logging.info("Launching TODO list MCP Server via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("MCP Server (stdio) stopped by user.")
    except Exception as e:
        logging.critical(f"MCP Server (stdio) encountered an unhandled error: {e}", exc_info=True)
    finally:
        logging.info("MCP Server (stdio) process exiting.")
