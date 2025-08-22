#!/usr/bin/env python3
"""
Pure MCP Tower Load Server
Simplified version without ADK dependencies for MCP client usage
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import mcp.server.stdio
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import network models and database utilities
try:
    from network_models import (
        TowerMetrics, CongestionEvent, SeverityLevel, TowerStatus,
        create_tower_metrics_from_db_row, create_congestion_event_from_db_row
    )
    from network_db_utils import NetworkDatabaseManager
except ImportError:
    # Fallback for when running as standalone MCP server
    import sys
    sys.path.append(os.path.dirname(__file__))
    from network_models import (
        TowerMetrics, CongestionEvent, SeverityLevel, TowerStatus,
        create_tower_metrics_from_db_row, create_congestion_event_from_db_row
    )
    from network_db_utils import NetworkDatabaseManager

load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Tower Load MCP] - %(message)s',
    handlers=[logging.StreamHandler()]
)

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")

class TowerLoadMonitor:
    """Core tower load monitoring functionality"""
    
    def __init__(self):
        self.db_manager = NetworkDatabaseManager()
        self.congestion_threshold = 80.0
        self.severity_thresholds = {
            'low': 80.0,
            'medium': 85.0,
            'high': 95.0
        }
        self.monitoring_active = False
        
    def monitor_tower_load(self, tower_id: str) -> Dict[str, Any]:
        """Monitor current load metrics for a specific tower"""
        try:
            # Get latest metrics
            metrics = self.db_manager.get_latest_tower_metrics(tower_id)
            if not metrics:
                return {
                    "success": False,
                    "message": f"No metrics found for tower {tower_id}",
                    "tower_id": tower_id
                }
            
            # Check if tower exists
            tower = self.db_manager.get_tower_by_id(tower_id)
            if not tower:
                return {
                    "success": False,
                    "message": f"Tower {tower_id} not found",
                    "tower_id": tower_id
                }
            
            result = {
                "success": True,
                "tower_id": tower_id,
                "tower_name": tower.name,
                "tower_status": tower.status.value,
                "metrics": metrics.to_dict(),
                "congestion_detected": metrics.is_congested,
                "congestion_severity": metrics.congestion_severity.value,
                "timestamp": metrics.timestamp.isoformat()
            }
            
            logging.info(f"Monitored tower {tower_id}: CPU={metrics.cpu_utilization}%, "
                        f"Memory={metrics.memory_usage}%, Bandwidth={metrics.bandwidth_usage}%")
            
            return result
            
        except Exception as e:
            logging.error(f"Error monitoring tower {tower_id}: {e}")
            return {
                "success": False,
                "message": f"Error monitoring tower {tower_id}: {str(e)}",
                "tower_id": tower_id
            }
    
    def detect_congestion(self, threshold: float = 80.0) -> Dict[str, Any]:
        """Detect congestion across all active towers"""
        try:
            self.congestion_threshold = threshold
            congested_towers = self.db_manager.get_congested_towers(threshold)
            
            congestion_results = []
            new_events_created = 0
            
            for tower_id, metrics in congested_towers:
                # Check if there's already an active congestion event
                active_events = self.db_manager.get_active_congestion_events()
                existing_event = next((e for e in active_events if e.tower_id == tower_id), None)
                
                if not existing_event:
                    # Create new congestion event
                    tower = self.db_manager.get_tower_by_id(tower_id)
                    affected_area = tower.name if tower else "Unknown"
                    
                    event = CongestionEvent(
                        tower_id=tower_id,
                        severity=metrics.congestion_severity,
                        detected_at=datetime.now(),
                        metrics=metrics,
                        affected_area=affected_area
                    )
                    
                    event_id = self.db_manager.insert_congestion_event(event)
                    event.id = event_id
                    new_events_created += 1
                    
                    logging.warning(f"New congestion event created for tower {tower_id} "
                                  f"with severity {metrics.congestion_severity.value}")
                
                congestion_results.append({
                    "tower_id": tower_id,
                    "severity": metrics.congestion_severity.value,
                    "cpu_utilization": metrics.cpu_utilization,
                    "memory_usage": metrics.memory_usage,
                    "bandwidth_usage": metrics.bandwidth_usage,
                    "active_connections": metrics.active_connections,
                    "detected_at": metrics.timestamp.isoformat(),
                    "existing_event": existing_event is not None
                })
            
            return {
                "success": True,
                "threshold_used": threshold,
                "total_congested_towers": len(congested_towers),
                "new_events_created": new_events_created,
                "congested_towers": congestion_results,
                "scan_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error detecting congestion: {e}")
            return {
                "success": False,
                "message": f"Error detecting congestion: {str(e)}",
                "threshold_used": threshold
            }
    
    def get_load_history(self, tower_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get historical load data for a tower"""
        try:
            metrics_history = self.db_manager.get_tower_metrics_history(tower_id, hours)
            
            if not metrics_history:
                return {
                    "success": False,
                    "message": f"No historical data found for tower {tower_id}",
                    "tower_id": tower_id,
                    "hours_requested": hours
                }
            
            # Calculate statistics
            cpu_values = [m.cpu_utilization for m in metrics_history]
            memory_values = [m.memory_usage for m in metrics_history]
            bandwidth_values = [m.bandwidth_usage for m in metrics_history]
            connection_values = [m.active_connections for m in metrics_history]
            
            # Count congestion periods
            congestion_periods = len([m for m in metrics_history if m.is_congested])
            
            # Find peak usage periods
            peak_cpu_metric = max(metrics_history, key=lambda m: m.cpu_utilization)
            peak_bandwidth_metric = max(metrics_history, key=lambda m: m.bandwidth_usage)
            
            result = {
                "success": True,
                "tower_id": tower_id,
                "period_hours": hours,
                "total_measurements": len(metrics_history),
                "statistics": {
                    "cpu_utilization": {
                        "average": round(sum(cpu_values) / len(cpu_values), 2),
                        "maximum": max(cpu_values),
                        "minimum": min(cpu_values),
                        "peak_timestamp": peak_cpu_metric.timestamp.isoformat()
                    },
                    "memory_usage": {
                        "average": round(sum(memory_values) / len(memory_values), 2),
                        "maximum": max(memory_values),
                        "minimum": min(memory_values)
                    },
                    "bandwidth_usage": {
                        "average": round(sum(bandwidth_values) / len(bandwidth_values), 2),
                        "maximum": max(bandwidth_values),
                        "minimum": min(bandwidth_values),
                        "peak_timestamp": peak_bandwidth_metric.timestamp.isoformat()
                    },
                    "active_connections": {
                        "average": round(sum(connection_values) / len(connection_values), 0),
                        "maximum": max(connection_values),
                        "minimum": min(connection_values)
                    }
                },
                "congestion_analysis": {
                    "total_congestion_periods": congestion_periods,
                    "congestion_percentage": round((congestion_periods / len(metrics_history)) * 100, 2)
                },
                "recent_metrics": [m.to_dict() for m in metrics_history[:5]],  # Last 5 measurements
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            logging.info(f"Retrieved {len(metrics_history)} historical metrics for tower {tower_id}")
            return result
            
        except Exception as e:
            logging.error(f"Error getting load history for tower {tower_id}: {e}")
            return {
                "success": False,
                "message": f"Error getting load history: {str(e)}",
                "tower_id": tower_id,
                "hours_requested": hours
            }

# Initialize tower load monitor
tower_monitor = TowerLoadMonitor()

# MCP Server Setup
logging.info("Creating Tower Load MCP Server instance...")
app = Server("tower-load-mcp-server")

@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    logging.info("Tower Load MCP Server: Received list_tools request.")
    
    tools = [
        mcp_types.Tool(
            name="monitor_tower_load",
            description="Monitor current load metrics for a specific tower",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "ID of the tower to monitor"
                    }
                },
                "required": ["tower_id"]
            }
        ),
        mcp_types.Tool(
            name="detect_congestion",
            description="Detect congestion across all active towers",
            inputSchema={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Congestion threshold percentage (default: 80.0)",
                        "default": 80.0
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="get_load_history",
            description="Get historical load data for a tower",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "ID of the tower"
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Number of hours of history to retrieve (default: 24)",
                        "default": 24
                    }
                },
                "required": ["tower_id"]
            }
        )
    ]
    
    return tools

@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """MCP handler to execute a tool call requested by an MCP client."""
    logging.info(f"Tower Load MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    try:
        if name == "monitor_tower_load":
            result = tower_monitor.monitor_tower_load(arguments.get("tower_id"))
        elif name == "detect_congestion":
            threshold = arguments.get("threshold", 80.0)
            result = tower_monitor.detect_congestion(threshold)
        elif name == "get_load_history":
            tower_id = arguments.get("tower_id")
            hours = arguments.get("hours", 24)
            result = tower_monitor.get_load_history(tower_id, hours)
        else:
            result = {
                "success": False,
                "message": f"Tool '{name}' not implemented by this server.",
                "available_tools": ["monitor_tower_load", "detect_congestion", "get_load_history"]
            }
        
        logging.info(f"Tower Load MCP Server: Tool '{name}' executed successfully")
        response_text = json.dumps(result, indent=2)
        return [mcp_types.TextContent(type="text", text=response_text)]

    except Exception as e:
        logging.error(f"Tower Load MCP Server: Error executing tool '{name}': {e}", exc_info=True)
        error_payload = {
            "success": False,
            "message": f"Failed to execute tool '{name}': {str(e)}",
            "tool_name": name
        }
        error_text = json.dumps(error_payload)
        return [mcp_types.TextContent(type="text", text=error_text)]

# MCP Server Runner
async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logging.info("Tower Load MCP Stdio Server: Starting handshake with client...")
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
        logging.info("Tower Load MCP Stdio Server: Run loop finished or client disconnected.")

if __name__ == "__main__":
    logging.info("Launching Tower Load MCP Server via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("\nTower Load MCP Server (stdio) stopped by user.")
    except Exception as e:
        logging.critical(f"Tower Load MCP Server (stdio) encountered an unhandled error: {e}", exc_info=True)
    finally:
        logging.info("Tower Load MCP Server (stdio) process exiting.")