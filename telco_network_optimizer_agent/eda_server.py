import asyncio
import json
import logging
import os
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression  # Example model
from sklearn.metrics import mean_squared_error  # For model evaluation

import mcp.server.stdio
from dotenv import load_dotenv
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

load_dotenv()

# # --- Logging Setup ---
# LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "mcp_server_activity.log")
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
#     handlers=[
#         logging.FileHandler(LOG_FILE_PATH, mode="w"),
#         logging.StreamHandler(),
#     ],
# )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output only
    ]
)    


# --- EDA Utility Functions ---
def get_dataset_overview(csv_file_path: str) -> dict:
    """Generates an overview of the dataset from a CSV file path."""
    try:
        data = pd.read_csv(csv_file_path)
        overview = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "head": data.head().to_dict(orient='records'),
        }
        return overview
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}

def check_data_quality(csv_file_path: str) -> dict:
    """Checks for missing values and data types from a CSV file path."""
    try:
        data = pd.read_csv(csv_file_path)
        missing_values = data.isnull().sum().to_dict()
        data_types = data.dtypes.astype(str).to_dict()
        return {
            "missing_values": missing_values,
            "data_types": data_types,
        }
    except Exception as e:
        return {"error": f"Failed to analyze data quality: {str(e)}"}

def suggest_insights(csv_file_path: str) -> dict:
    """Suggests insights based on the dataset from a CSV file path."""
    try:
        data = pd.read_csv(csv_file_path)
        insights = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            insights[column] = {
                "mean": float(data[column].mean()) if not pd.isna(data[column].mean()) else None,
                "median": float(data[column].median()) if not pd.isna(data[column].median()) else None,
                "std_dev": float(data[column].std()) if not pd.isna(data[column].std()) else None,
            }
        return insights
    except Exception as e:
        return {"error": f"Failed to generate insights: {str(e)}"}

def visualize_column(csv_file_path: str, column: str) -> str:
    """Generates a visualization for a specific column from a CSV file path."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        data = pd.read_csv(csv_file_path)
        if column not in data.columns:
            return f"Error: Column '{column}' not found in dataset"
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        output_file = f'{column}_distribution.png'
        plt.savefig(output_file)
        plt.close()
        return f"Visualization saved as {output_file}"
    except Exception as e:
        return f"Error generating visualization: {str(e)}"

def run_correlation_analysis(csv_file_path: str) -> dict:
    """Runs correlation analysis on numerical columns from a CSV file path."""
    try:
        data = pd.read_csv(csv_file_path)
        numerical_data = data.select_dtypes(include=[np.number])
        if numerical_data.empty:
            return {"error": "No numerical columns found for correlation analysis"}
        
        correlation_matrix = numerical_data.corr().to_dict()
        return correlation_matrix
    except Exception as e:
        return {"error": f"Failed to run correlation analysis: {str(e)}"}

def recommend_model_type(csv_file_path: str, target_column: str) -> str:
    """Recommends a model type based on the target variable from a CSV file path."""
    try:
        data = pd.read_csv(csv_file_path)
        if target_column not in data.columns:
            return f"Error: Target column '{target_column}' not found in dataset"
        
        if data[target_column].dtype == 'object':
            return "Consider using classification models (target variable appears to be categorical)."
        else:
            return "Consider using regression models (target variable appears to be numerical)."
    except Exception as e:
        return f"Error analyzing target column: {str(e)}"

# --- MCP Server Setup ---
logging.info("Creating EDA MCP Server instance...")
from mcp.server import Server
app = Server("eda-server")

# Wrap EDA utility functions as ADK FunctionTools
ADK_EDA_TOOLS = {
    "get_dataset_overview": FunctionTool(func=get_dataset_overview),
    "check_data_quality": FunctionTool(func=check_data_quality),
    "suggest_insights": FunctionTool(func=suggest_insights),
    "visualize_column": FunctionTool(func=visualize_column),
    "run_correlation_analysis": FunctionTool(func=run_correlation_analysis),
    "recommend_model_type": FunctionTool(func=recommend_model_type),
}

from mcp import types as mcp_types

# Create a simple notification options object
class NotificationOptions:
    def __init__(self):
        self.tools_changed = False
        self.prompts_changed = False
        self.resources_changed = False

@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    logging.info("MCP Server: Received list_tools request.")
    mcp_tools_list = []
    for tool_name, adk_tool_instance in ADK_EDA_TOOLS.items():
        if not adk_tool_instance.name:
            adk_tool_instance.name = tool_name

        mcp_tool_schema = adk_to_mcp_tool_type(adk_tool_instance)
        logging.info(f"MCP Server: Advertising tool: {mcp_tool_schema.name}, InputSchema: {mcp_tool_schema.inputSchema}")
        mcp_tools_list.append(mcp_tool_schema)
    return mcp_tools_list

@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """MCP handler to execute a tool call requested by an MCP client."""
    logging.info(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    if name in ADK_EDA_TOOLS:
        adk_tool_instance = ADK_EDA_TOOLS[name]
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
            error_text = json.dumps(error_payload)
            return [mcp_types.TextContent(type="text", text=error_text)]
    else:
        logging.warning(f"MCP Server: Tool '{name}' not found/exposed by this server.")
        error_payload = {
            "success": False,
            "message": f"Tool '{name}' not implemented by this server.",
        }
        error_text = json.dumps(error_payload)
        return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Server Runner ---
from mcp.server.models import InitializationOptions

async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output.""" 
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
                    experimental_capabilities={}
                ),
            ),
        )
        logging.info("MCP Stdio Server: Run loop finished or client disconnected.")

if __name__ == "__main__":
    logging.info("Launching EDA MCP Server via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("\nMCP Server (stdio) stopped by user.")
    except Exception as e:
        logging.critical(f"MCP Server (stdio) encountered an unhandled error: {e}", exc_info=True)
    finally:
        logging.info("MCP Server (stdio) process exiting.")