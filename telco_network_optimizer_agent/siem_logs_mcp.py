# siem_mcp_server.py
import asyncio
import logging
import json
from datetime import datetime
from google.adk.tools.mcp_tool.mcp_toolset import McpToolServer

# Assuming these are in a 'telco_network_optimizer_agent' directory or otherwise importable
from siem_logs_mcp import SIEMLogsMCP, ThreatCriteria, SuspiciousActivity, UserEvent, PortScanEvent
from models import SecurityEvent, ThreatSeverity # Assuming models.py defines these

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SIEMLogsMcpToolServer:
    """McpToolServer wrapper for SIEMLogsMCP functionalities."""

    def __init__(self):
        # Initialize SIEMLogsMCP with a basic config. In a real scenario, this would be more robust.
        self.siem_mcp = SIEMLogsMCP(config={'siem_type': 'MockSIEM', 'port_scan_threshold': 60})
        logger.info("SIEMLogsMcpToolServer initialized with SIEMLogsMCP.")

    async def connect(self) -> bool:
        """Establishes connection to SIEM systems."""
        logger.info("Attempting to connect SIEMLogsMCP.")
        return await self.siem_mcp.connect()

    async def disconnect(self):
        """Disconnects from SIEM systems."""
        logger.info("Attempting to disconnect SIEMLogsMCP.")
        await self.siem_mcp.disconnect()

    async def get_recent_events(self, time_window: int, event_types: list) -> list:
        """
        Get recent security events from SIEM logs.
        Args:
            time_window: Time window in minutes to look back.
            event_types: List of event types to filter for.
        Returns:
            List of SecurityEvent objects (serialized as dicts).
        """
        logger.info(f"SIEMLogsMcpToolServer: get_recent_events called with time_window={time_window}, event_types={event_types}")
        events = await self.siem_mcp.get_recent_events(time_window, event_types)
        # Convert dataclass instances to dictionaries for JSON serialization
        return [self._serialize_dataclass(event) for event in events]

    async def query_suspicious_activity(self, criteria: dict) -> list:
        """
        Query for suspicious activity based on criteria.
        Args:
            criteria: Dictionary defining ThreatCriteria parameters.
        Returns:
            List of SuspiciousActivity objects (serialized as dicts).
        """
        logger.info(f"SIEMLogsMcpToolServer: query_suspicious_activity called with criteria={criteria}")
        # Convert dictionary to ThreatCriteria dataclass
        try:
            threat_criteria = ThreatCriteria(
                event_types=criteria['event_types'],
                severity_threshold=ThreatSeverity[criteria['severity_threshold'].upper()], # Convert string to Enum
                time_window_hours=criteria['time_window_hours'],
                source_ips=criteria.get('source_ips'),
                user_ids=criteria.get('user_ids')
            )
        except KeyError as e:
            logger.error(f"Missing key in criteria for ThreatCriteria: {e}")
            raise ValueError(f"Invalid criteria for query_suspicious_activity: Missing {e}")
        except Exception as e:
            logger.error(f"Error converting criteria to ThreatCriteria: {e}")
            raise

        activities = await self.siem_mcp.query_suspicious_activity(threat_criteria)
        return [self._serialize_dataclass(activity) for activity in activities]

    async def get_user_activity_timeline(self, user_id: str, hours: int) -> list:
        """
        Get user activity timeline for analysis.
        Args:
            user_id: User identifier to analyze.
            hours: Number of hours to look back.
        Returns:
            List of UserEvent objects (serialized as dicts).
        """
        logger.info(f"SIEMLogsMcpToolServer: get_user_activity_timeline called for user_id={user_id}, hours={hours}")
        events = await self.siem_mcp.get_user_activity_timeline(user_id, hours)
        return [self._serialize_dataclass(event) for event in events]

    async def detect_port_scanning(self, threshold: int) -> list:
        """
        Detect port scanning activity in network logs.
        Args:
            threshold: Minimum number of port attempts to consider as scanning.
        Returns:
            List of PortScanEvent objects (serialized as dicts).
        """
        logger.info(f"SIEMLogsMcpToolServer: detect_port_scanning called with threshold={threshold}")
        scans = await self.siem_mcp.detect_port_scanning(threshold)
        return [self._serialize_dataclass(scan) for scan in scans]

    async def health_check(self) -> dict:
        """Perform health check on SIEM connections."""
        logger.info("SIEMLogsMcpToolServer: health_check called.")
        health_status = await self.siem_mcp.health_check()
        return health_status

    def _serialize_dataclass(self, instance):
        """Helper to serialize dataclass instances, handling datetime and Enum."""
        serialized_data = {}
        for field in instance.__dataclass_fields__:
            value = getattr(instance, field)
            if isinstance(value, datetime):
                serialized_data[field] = value.isoformat()
            elif isinstance(value, ThreatSeverity):
                serialized_data[field] = value.name # Convert Enum to its name string
            elif isinstance(value, list) and all(isinstance(item, (datetime, ThreatSeverity)) for item in value):
                # Handle lists containing datetimes or Enums
                serialized_data[field] = [item.isoformat() if isinstance(item, datetime) else item.name for item in value]
            elif hasattr(value, '__dataclass_fields__'): # Nested dataclasses
                serialized_data[field] = self._serialize_dataclass(value)
            else:
                serialized_data[field] = value
        return serialized_data

async def main():
    siem_tool_server_instance = SIEMLogsMcpToolServer()
    server = McpToolServer(siem_tool_server_instance)
    logger.info("Starting SIEM MCP Tool Server to listen for requests...")
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())