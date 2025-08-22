"""
Maintenance MCP Server for Telecom Network Optimization
Handles hardware monitoring, maintenance ticket management, and predictive maintenance
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import mcp.server.stdio
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
    handlers=[
        logging.StreamHandler(),
    ]
)

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")

# --- Data Models ---

class ComponentType(Enum):
    ANTENNA = "ANTENNA"
    PROCESSOR = "PROCESSOR"
    MEMORY = "MEMORY"
    POWER = "POWER"
    COOLING = "COOLING"
    NETWORK = "NETWORK"

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"

class Priority(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class TicketStatus(Enum):
    OPEN = "OPEN"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"

@dataclass
class HardwareStatus:
    tower_id: str
    component_type: str
    status: str
    last_checked: str
    error_codes: Optional[str] = None
    performance_metrics: Optional[str] = None
    temperature: Optional[float] = None

@dataclass
class MaintenanceTicket:
    tower_id: str
    priority: str
    issue_description: str
    created_at: str
    assigned_to: Optional[str] = None
    status: str = "OPEN"
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None

# --- Database Utility Functions ---

def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_maintenance_schedules_table():
    """Ensure the maintenance_schedules table exists for predictive maintenance"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create maintenance_schedules table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id INTEGER NOT NULL,
                tower_id TEXT NOT NULL,
                component_type TEXT NOT NULL,
                scheduled_date DATETIME NOT NULL,
                maintenance_type TEXT NOT NULL CHECK(maintenance_type IN ('PROACTIVE', 'REACTIVE', 'PREVENTIVE', 'EMERGENCY')),
                priority TEXT NOT NULL CHECK(priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                status TEXT DEFAULT 'SCHEDULED' CHECK(status IN ('SCHEDULED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED', 'POSTPONED')),
                assigned_technician TEXT,
                estimated_duration_hours REAL DEFAULT 2.0,
                actual_start_time DATETIME,
                actual_end_time DATETIME,
                completion_notes TEXT,
                cost_estimate REAL,
                actual_cost REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES maintenance_tickets(id),
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        
        # Create index for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_maintenance_schedules_tower_date 
            ON maintenance_schedules(tower_id, scheduled_date)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_maintenance_schedules_status 
            ON maintenance_schedules(status, priority)
        """)
        
        conn.commit()
        conn.close()
        
    except sqlite3.Error as e:
        logging.error(f"Error creating maintenance_schedules table: {e}")
        raise

# Initialize the table when module is imported
ensure_maintenance_schedules_table()

def monitor_hardware_health(tower_id: str) -> Dict[str, Any]:
    """
    Monitor hardware health for a specific tower
    
    Args:
        tower_id: The ID of the tower to monitor
        
    Returns:
        Dictionary containing hardware status information
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get latest hardware status for all components of the tower
        cursor.execute("""
            SELECT tower_id, component_type, status, last_checked, 
                   error_codes, performance_metrics, temperature
            FROM hardware_status 
            WHERE tower_id = ? 
            ORDER BY component_type, last_checked DESC
        """, (tower_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {
                "success": False,
                "message": f"No hardware status found for tower {tower_id}",
                "hardware_status": []
            }
        
        # Group by component type and get latest status
        hardware_status = {}
        for row in results:
            component = row["component_type"]
            if component not in hardware_status:
                hardware_status[component] = {
                    "tower_id": row["tower_id"],
                    "component_type": component,
                    "status": row["status"],
                    "last_checked": row["last_checked"],
                    "error_codes": row["error_codes"],
                    "performance_metrics": row["performance_metrics"],
                    "temperature": row["temperature"]
                }
        
        # Calculate overall health score
        status_weights = {"HEALTHY": 100, "WARNING": 70, "CRITICAL": 30, "FAILED": 0}
        total_score = sum(status_weights.get(comp["status"], 0) for comp in hardware_status.values())
        avg_score = total_score / len(hardware_status) if hardware_status else 0
        
        overall_status = "HEALTHY"
        if avg_score < 25:
            overall_status = "FAILED"
        elif avg_score < 50:
            overall_status = "CRITICAL"
        elif avg_score < 80:
            overall_status = "WARNING"
        
        return {
            "success": True,
            "message": f"Hardware health retrieved for tower {tower_id}",
            "tower_id": tower_id,
            "overall_status": overall_status,
            "health_score": round(avg_score, 2),
            "components": list(hardware_status.values()),
            "component_count": len(hardware_status)
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error monitoring hardware health: {e}",
            "hardware_status": []
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error monitoring hardware health: {e}",
            "hardware_status": []
        }

def detect_hardware_anomalies(threshold_hours: int = 24) -> Dict[str, Any]:
    """
    Detect hardware anomalies across all towers within the specified time window
    
    Args:
        threshold_hours: Hours to look back for anomaly detection
        
    Returns:
        Dictionary containing detected anomalies
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate threshold timestamp
        threshold_time = datetime.now() - timedelta(hours=threshold_hours)
        
        # Find components with WARNING, CRITICAL, or FAILED status
        cursor.execute("""
            SELECT tower_id, component_type, status, last_checked, 
                   error_codes, performance_metrics, temperature
            FROM hardware_status 
            WHERE status IN ('WARNING', 'CRITICAL', 'FAILED')
            AND last_checked >= ?
            ORDER BY 
                CASE status 
                    WHEN 'FAILED' THEN 1 
                    WHEN 'CRITICAL' THEN 2 
                    WHEN 'WARNING' THEN 3 
                END,
                last_checked DESC
        """, (threshold_time.isoformat(),))
        
        anomalies = []
        for row in cursor.fetchall():
            anomaly = {
                "tower_id": row["tower_id"],
                "component_type": row["component_type"],
                "status": row["status"],
                "last_checked": row["last_checked"],
                "error_codes": row["error_codes"],
                "performance_metrics": row["performance_metrics"],
                "temperature": row["temperature"],
                "severity_score": {"FAILED": 100, "CRITICAL": 80, "WARNING": 40}.get(row["status"], 0)
            }
            anomalies.append(anomaly)
        
        # Get tower names for better reporting
        if anomalies:
            tower_ids = list(set(a["tower_id"] for a in anomalies))
            placeholders = ",".join("?" * len(tower_ids))
            cursor.execute(f"SELECT id, name FROM towers WHERE id IN ({placeholders})", tower_ids)
            tower_names = {row["id"]: row["name"] for row in cursor.fetchall()}
            
            for anomaly in anomalies:
                anomaly["tower_name"] = tower_names.get(anomaly["tower_id"], "Unknown")
        
        conn.close()
        
        # Categorize anomalies by severity
        critical_count = len([a for a in anomalies if a["status"] in ["FAILED", "CRITICAL"]])
        warning_count = len([a for a in anomalies if a["status"] == "WARNING"])
        
        return {
            "success": True,
            "message": f"Detected {len(anomalies)} hardware anomalies in the last {threshold_hours} hours",
            "anomaly_count": len(anomalies),
            "critical_count": critical_count,
            "warning_count": warning_count,
            "anomalies": anomalies,
            "threshold_hours": threshold_hours
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error detecting anomalies: {e}",
            "anomalies": []
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error detecting anomalies: {e}",
            "anomalies": []
        }

def update_hardware_status(tower_id: str, component_type: str, status: str, 
                          error_codes: str = "", performance_metrics: str = "", 
                          temperature: float = None) -> Dict[str, Any]:
    """
    Update hardware status for a specific tower component
    
    Args:
        tower_id: The ID of the tower
        component_type: Type of component (ANTENNA, PROCESSOR, etc.)
        status: Health status (HEALTHY, WARNING, CRITICAL, FAILED)
        error_codes: Optional error codes
        performance_metrics: Optional performance metrics JSON
        temperature: Optional temperature reading
        
    Returns:
        Dictionary indicating success/failure
    """
    try:
        # Validate inputs
        if component_type not in [e.value for e in ComponentType]:
            return {
                "success": False,
                "message": f"Invalid component_type. Must be one of: {[e.value for e in ComponentType]}"
            }
        
        if status not in [e.value for e in HealthStatus]:
            return {
                "success": False,
                "message": f"Invalid status. Must be one of: {[e.value for e in HealthStatus]}"
            }
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert or update hardware status
        cursor.execute("""
            INSERT INTO hardware_status 
            (tower_id, component_type, status, last_checked, error_codes, performance_metrics, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            tower_id, component_type, status, datetime.now().isoformat(),
            error_codes or None, performance_metrics or None, temperature
        ))
        
        conn.commit()
        status_id = cursor.lastrowid
        conn.close()
        
        return {
            "success": True,
            "message": f"Hardware status updated for {tower_id} {component_type}",
            "status_id": status_id,
            "tower_id": tower_id,
            "component_type": component_type,
            "status": status
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error updating hardware status: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error updating hardware status: {e}"
        }

def get_equipment_status_summary(dummy_param: str = "") -> Dict[str, Any]:
    """
    Get a summary of equipment status across all towers
    
    Args:
        dummy_param: Unused parameter for schema generation
        
    Returns:
        Dictionary containing equipment status summary
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get latest status for each tower-component combination
        cursor.execute("""
            WITH latest_status AS (
                SELECT tower_id, component_type, status, last_checked,
                       ROW_NUMBER() OVER (PARTITION BY tower_id, component_type ORDER BY last_checked DESC) as rn
                FROM hardware_status
            )
            SELECT tower_id, component_type, status, last_checked
            FROM latest_status 
            WHERE rn = 1
            ORDER BY tower_id, component_type
        """)
        
        results = cursor.fetchall()
        
        # Get tower information
        cursor.execute("SELECT id, name, status FROM towers ORDER BY id")
        towers = {row["id"]: {"name": row["name"], "tower_status": row["status"]} for row in cursor.fetchall()}
        
        conn.close()
        
        # Organize data by tower
        tower_status = {}
        status_counts = {"HEALTHY": 0, "WARNING": 0, "CRITICAL": 0, "FAILED": 0}
        
        for row in results:
            tower_id = row["tower_id"]
            if tower_id not in tower_status:
                tower_status[tower_id] = {
                    "tower_id": tower_id,
                    "tower_name": towers.get(tower_id, {}).get("name", "Unknown"),
                    "tower_status": towers.get(tower_id, {}).get("tower_status", "UNKNOWN"),
                    "components": {},
                    "health_summary": {"HEALTHY": 0, "WARNING": 0, "CRITICAL": 0, "FAILED": 0}
                }
            
            component_type = row["component_type"]
            status = row["status"]
            
            tower_status[tower_id]["components"][component_type] = {
                "status": status,
                "last_checked": row["last_checked"]
            }
            tower_status[tower_id]["health_summary"][status] += 1
            status_counts[status] += 1
        
        # Calculate overall health scores
        for tower_data in tower_status.values():
            total_components = sum(tower_data["health_summary"].values())
            if total_components > 0:
                health_score = (
                    tower_data["health_summary"]["HEALTHY"] * 100 +
                    tower_data["health_summary"]["WARNING"] * 70 +
                    tower_data["health_summary"]["CRITICAL"] * 30 +
                    tower_data["health_summary"]["FAILED"] * 0
                ) / total_components
                tower_data["health_score"] = round(health_score, 2)
            else:
                tower_data["health_score"] = 0
        
        return {
            "success": True,
            "message": f"Equipment status summary retrieved for {len(tower_status)} towers",
            "total_towers": len(tower_status),
            "overall_status_counts": status_counts,
            "tower_details": list(tower_status.values())
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error getting equipment status: {e}",
            "tower_details": []
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error getting equipment status: {e}",
            "tower_details": []
        }

def create_maintenance_ticket(tower_id: str, issue_description: str, priority: str = "MEDIUM", 
                            assigned_to: str = None) -> Dict[str, Any]:
    """
    Create a new maintenance ticket for a tower issue
    
    Args:
        tower_id: The ID of the tower with the issue
        issue_description: Description of the maintenance issue
        priority: Priority level (LOW, MEDIUM, HIGH, CRITICAL)
        assigned_to: Optional technician assignment
        
    Returns:
        Dictionary containing ticket creation result
    """
    try:
        # Validate priority
        if priority not in [e.value for e in Priority]:
            return {
                "success": False,
                "message": f"Invalid priority. Must be one of: {[e.value for e in Priority]}"
            }
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify tower exists
        cursor.execute("SELECT id, name FROM towers WHERE id = ?", (tower_id,))
        tower = cursor.fetchone()
        if not tower:
            conn.close()
            return {
                "success": False,
                "message": f"Tower {tower_id} not found"
            }
        
        # Create maintenance ticket
        cursor.execute("""
            INSERT INTO maintenance_tickets 
            (tower_id, priority, issue_description, created_at, assigned_to, status)
            VALUES (?, ?, ?, ?, ?, 'OPEN')
        """, (
            tower_id, priority, issue_description, 
            datetime.now().isoformat(), assigned_to
        ))
        
        ticket_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Maintenance ticket created for tower {tower_id}",
            "ticket_id": ticket_id,
            "tower_id": tower_id,
            "tower_name": tower["name"],
            "priority": priority,
            "status": "OPEN",
            "assigned_to": assigned_to,
            "issue_description": issue_description
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error creating maintenance ticket: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error creating maintenance ticket: {e}"
        }

def get_maintenance_tickets(tower_id: str = None, status: str = None, 
                          priority: str = None, limit: int = 50) -> Dict[str, Any]:
    """
    Retrieve maintenance tickets with optional filtering
    
    Args:
        tower_id: Optional filter by tower ID
        status: Optional filter by ticket status
        priority: Optional filter by priority level
        limit: Maximum number of tickets to return
        
    Returns:
        Dictionary containing filtered maintenance tickets
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query with filters
        query = """
            SELECT mt.id, mt.tower_id, t.name as tower_name, mt.priority, 
                   mt.issue_description, mt.created_at, mt.assigned_to, 
                   mt.status, mt.resolved_at, mt.resolution_notes
            FROM maintenance_tickets mt
            LEFT JOIN towers t ON mt.tower_id = t.id
            WHERE 1=1
        """
        params = []
        
        if tower_id:
            query += " AND mt.tower_id = ?"
            params.append(tower_id)
        
        if status:
            if status not in [e.value for e in TicketStatus]:
                conn.close()
                return {
                    "success": False,
                    "message": f"Invalid status. Must be one of: {[e.value for e in TicketStatus]}"
                }
            query += " AND mt.status = ?"
            params.append(status)
        
        if priority:
            if priority not in [e.value for e in Priority]:
                conn.close()
                return {
                    "success": False,
                    "message": f"Invalid priority. Must be one of: {[e.value for e in Priority]}"
                }
            query += " AND mt.priority = ?"
            params.append(priority)
        
        query += " ORDER BY mt.created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        tickets = []
        for row in results:
            ticket = {
                "ticket_id": row["id"],
                "tower_id": row["tower_id"],
                "tower_name": row["tower_name"] or "Unknown",
                "priority": row["priority"],
                "issue_description": row["issue_description"],
                "created_at": row["created_at"],
                "assigned_to": row["assigned_to"],
                "status": row["status"],
                "resolved_at": row["resolved_at"],
                "resolution_notes": row["resolution_notes"]
            }
            tickets.append(ticket)
        
        # Get summary statistics
        cursor = get_db_connection().cursor()
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM maintenance_tickets 
            GROUP BY status
        """)
        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT priority, COUNT(*) as count 
            FROM maintenance_tickets 
            GROUP BY priority
        """)
        priority_counts = {row["priority"]: row["count"] for row in cursor.fetchall()}
        cursor.close()
        
        return {
            "success": True,
            "message": f"Retrieved {len(tickets)} maintenance tickets",
            "ticket_count": len(tickets),
            "tickets": tickets,
            "status_summary": status_counts,
            "priority_summary": priority_counts,
            "filters_applied": {
                "tower_id": tower_id,
                "status": status,
                "priority": priority,
                "limit": limit
            }
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error retrieving maintenance tickets: {e}",
            "tickets": []
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error retrieving maintenance tickets: {e}",
            "tickets": []
        }

def update_maintenance_ticket(ticket_id: int, status: str = None, assigned_to: str = None, 
                            resolution_notes: str = None) -> Dict[str, Any]:
    """
    Update an existing maintenance ticket
    
    Args:
        ticket_id: ID of the ticket to update
        status: New status (OPEN, ASSIGNED, IN_PROGRESS, RESOLVED, CLOSED)
        assigned_to: Technician to assign the ticket to
        resolution_notes: Notes about the resolution
        
    Returns:
        Dictionary containing update result
    """
    try:
        if status and status not in [e.value for e in TicketStatus]:
            return {
                "success": False,
                "message": f"Invalid status. Must be one of: {[e.value for e in TicketStatus]}"
            }
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if ticket exists
        cursor.execute("SELECT * FROM maintenance_tickets WHERE id = ?", (ticket_id,))
        ticket = cursor.fetchone()
        if not ticket:
            conn.close()
            return {
                "success": False,
                "message": f"Maintenance ticket {ticket_id} not found"
            }
        
        # Build update query
        updates = []
        params = []
        
        if status:
            updates.append("status = ?")
            params.append(status)
            
            # Set resolved_at timestamp if status is RESOLVED or CLOSED
            if status in ["RESOLVED", "CLOSED"]:
                updates.append("resolved_at = ?")
                params.append(datetime.now().isoformat())
        
        if assigned_to is not None:  # Allow empty string to unassign
            updates.append("assigned_to = ?")
            params.append(assigned_to if assigned_to else None)
        
        if resolution_notes:
            updates.append("resolution_notes = ?")
            params.append(resolution_notes)
        
        if not updates:
            conn.close()
            return {
                "success": False,
                "message": "No updates provided"
            }
        
        # Execute update
        params.append(ticket_id)
        query = f"UPDATE maintenance_tickets SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        
        # Get updated ticket
        cursor.execute("""
            SELECT mt.*, t.name as tower_name 
            FROM maintenance_tickets mt
            LEFT JOIN towers t ON mt.tower_id = t.id
            WHERE mt.id = ?
        """, (ticket_id,))
        updated_ticket = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Maintenance ticket {ticket_id} updated successfully",
            "ticket_id": ticket_id,
            "tower_id": updated_ticket["tower_id"],
            "tower_name": updated_ticket["tower_name"] or "Unknown",
            "priority": updated_ticket["priority"],
            "status": updated_ticket["status"],
            "assigned_to": updated_ticket["assigned_to"],
            "resolved_at": updated_ticket["resolved_at"],
            "resolution_notes": updated_ticket["resolution_notes"]
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error updating maintenance ticket: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error updating maintenance ticket: {e}"
        }

def classify_issue_priority(hardware_status: Dict[str, Any], error_codes: str = None, 
                          temperature: float = None, performance_metrics: str = None) -> Dict[str, Any]:
    """
    Enhanced priority classification algorithm for maintenance issues
    
    Args:
        hardware_status: Hardware status information
        error_codes: Optional error codes from the hardware
        temperature: Optional temperature reading
        performance_metrics: Optional performance metrics JSON
        
    Returns:
        Dictionary containing priority classification result
    """
    try:
        status = hardware_status.get("status", "HEALTHY").upper()
        component_type = hardware_status.get("component_type", "").upper()
        
        # Initialize priority score (0-100, higher = more critical)
        priority_score = 0
        classification_factors = []
        
        # Base priority from status
        status_scores = {
            "FAILED": 90,
            "CRITICAL": 70, 
            "WARNING": 40,
            "HEALTHY": 10
        }
        priority_score += status_scores.get(status, 10)
        classification_factors.append(f"Status: {status} (+{status_scores.get(status, 10)})")
        
        # Component criticality multiplier
        critical_components = {
            "POWER": 1.5,      # Power failures affect entire tower
            "COOLING": 1.4,    # Overheating can cascade to other components
            "PROCESSOR": 1.3,  # Core processing affects all services
            "ANTENNA": 1.2,    # Direct impact on signal quality
            "NETWORK": 1.1,    # Connectivity issues
            "MEMORY": 1.0      # Less critical but still important
        }
        
        multiplier = critical_components.get(component_type, 1.0)
        priority_score = int(priority_score * multiplier)
        classification_factors.append(f"Component: {component_type} (x{multiplier})")
        
        # Temperature factor
        if temperature is not None:
            if temperature > 80:  # Critical temperature
                priority_score += 20
                classification_factors.append(f"Temperature: {temperature}°C (+20)")
            elif temperature > 70:  # Warning temperature
                priority_score += 10
                classification_factors.append(f"Temperature: {temperature}°C (+10)")
        
        # Error code severity analysis
        if error_codes:
            error_severity = analyze_error_codes(error_codes)
            priority_score += error_severity
            classification_factors.append(f"Error codes: {error_codes} (+{error_severity})")
        
        # Performance metrics analysis
        if performance_metrics:
            try:
                import json
                metrics = json.loads(performance_metrics)
                perf_impact = analyze_performance_impact(metrics)
                priority_score += perf_impact
                classification_factors.append(f"Performance impact (+{perf_impact})")
            except (json.JSONDecodeError, Exception):
                pass
        
        # Determine final priority level
        if priority_score >= 85:
            priority = "CRITICAL"
        elif priority_score >= 65:
            priority = "HIGH"
        elif priority_score >= 35:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        return {
            "success": True,
            "priority": priority,
            "priority_score": min(priority_score, 100),
            "classification_factors": classification_factors,
            "component_type": component_type,
            "status": status,
            "recommendation": get_priority_recommendation(priority, component_type)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Priority classification failed: {e}",
            "priority": "MEDIUM",  # Safe default
            "priority_score": 50
        }

def predict_hardware_failure(tower_id: str, component_type: str = None, 
                           prediction_horizon_hours: int = 168) -> Dict[str, Any]:
    """
    Predict hardware failure probability using historical data and trends
    
    Args:
        tower_id: The ID of the tower to analyze
        component_type: Optional specific component to analyze
        prediction_horizon_hours: Hours ahead to predict (default: 168 = 1 week)
        
    Returns:
        Dictionary containing failure prediction results
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get historical hardware status data for trend analysis
        base_query = """
            SELECT component_type, status, last_checked, error_codes, 
                   performance_metrics, temperature
            FROM hardware_status 
            WHERE tower_id = ?
        """
        params = [tower_id]
        
        if component_type:
            base_query += " AND component_type = ?"
            params.append(component_type)
        
        base_query += " ORDER BY component_type, last_checked DESC"
        
        cursor.execute(base_query, params)
        historical_data = cursor.fetchall()
        
        if not historical_data:
            conn.close()
            return {
                "success": False,
                "message": f"No historical data found for tower {tower_id}",
                "predictions": []
            }
        
        # Group data by component type
        component_data = {}
        for row in historical_data:
            comp_type = row["component_type"]
            if comp_type not in component_data:
                component_data[comp_type] = []
            component_data[comp_type].append(row)
        
        predictions = []
        
        for comp_type, data_points in component_data.items():
            # Analyze failure patterns and trends
            failure_prediction = analyze_component_failure_risk(comp_type, data_points, prediction_horizon_hours)
            predictions.append(failure_prediction)
        
        # Calculate overall tower failure risk
        if predictions:
            avg_failure_prob = sum(p["failure_probability"] for p in predictions) / len(predictions)
            max_failure_prob = max(p["failure_probability"] for p in predictions)
            
            # Determine overall risk level
            if max_failure_prob >= 0.8:
                overall_risk = "CRITICAL"
            elif max_failure_prob >= 0.6:
                overall_risk = "HIGH"
            elif max_failure_prob >= 0.4:
                overall_risk = "MEDIUM"
            else:
                overall_risk = "LOW"
        else:
            avg_failure_prob = 0.0
            max_failure_prob = 0.0
            overall_risk = "LOW"
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Hardware failure prediction completed for tower {tower_id}",
            "tower_id": tower_id,
            "prediction_horizon_hours": prediction_horizon_hours,
            "overall_risk_level": overall_risk,
            "average_failure_probability": round(avg_failure_prob, 3),
            "maximum_failure_probability": round(max_failure_prob, 3),
            "component_predictions": predictions,
            "recommendations": generate_predictive_maintenance_recommendations(predictions)
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error predicting hardware failure: {e}",
            "predictions": []
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error predicting hardware failure: {e}",
            "predictions": []
        }

def analyze_component_failure_risk(component_type: str, data_points: List, 
                                 prediction_horizon_hours: int) -> Dict[str, Any]:
    """
    Analyze failure risk for a specific component based on historical data
    
    Args:
        component_type: Type of component to analyze
        data_points: Historical data points for the component
        prediction_horizon_hours: Prediction time horizon
        
    Returns:
        Dictionary containing component failure risk analysis
    """
    try:
        # Calculate degradation trends
        status_weights = {"HEALTHY": 1.0, "WARNING": 0.7, "CRITICAL": 0.3, "FAILED": 0.0}
        
        # Analyze status trend over time
        status_trend = []
        temperature_trend = []
        error_frequency = 0
        
        for i, point in enumerate(data_points[:10]):  # Last 10 data points
            status_score = status_weights.get(point["status"], 0.5)
            status_trend.append(status_score)
            
            if point["temperature"]:
                temperature_trend.append(point["temperature"])
            
            if point["error_codes"]:
                error_frequency += 1
        
        # Calculate trend slope (negative slope indicates degradation)
        if len(status_trend) >= 2:
            trend_slope = calculate_trend_slope(status_trend)
        else:
            trend_slope = 0.0
        
        # Component-specific failure risk factors
        component_risk_factors = {
            "POWER": {"base_risk": 0.15, "temp_threshold": 60, "critical_errors": ["CRIT_", "FATAL_"]},
            "COOLING": {"base_risk": 0.20, "temp_threshold": 70, "critical_errors": ["OVERHEAT", "FAN_FAIL"]},
            "PROCESSOR": {"base_risk": 0.10, "temp_threshold": 80, "critical_errors": ["CPU_", "PROC_"]},
            "ANTENNA": {"base_risk": 0.08, "temp_threshold": 65, "critical_errors": ["RF_", "SIG_"]},
            "MEMORY": {"base_risk": 0.05, "temp_threshold": 75, "critical_errors": ["MEM_", "RAM_"]},
            "NETWORK": {"base_risk": 0.12, "temp_threshold": 70, "critical_errors": ["NET_", "CONN_"]}
        }
        
        risk_factors = component_risk_factors.get(component_type, {"base_risk": 0.10, "temp_threshold": 70, "critical_errors": []})
        
        # Calculate failure probability
        base_risk = risk_factors["base_risk"]
        
        # Trend factor (degrading trend increases risk significantly)
        trend_factor = max(0, -trend_slope * 2.0)  # Increased sensitivity to negative trends
        
        # Temperature factor
        temp_factor = 0.0
        if temperature_trend:
            avg_temp = sum(temperature_trend) / len(temperature_trend)
            if avg_temp > risk_factors["temp_threshold"]:
                temp_factor = min(0.5, (avg_temp - risk_factors["temp_threshold"]) / 50)  # More sensitive
        
        # Error frequency factor (more weight to frequent errors)
        error_factor = min(0.6, error_frequency / len(data_points) * 2.0)
        
        # Status degradation factor (check recent status changes)
        status_degradation_factor = 0.0
        if len(status_trend) >= 3:
            recent_avg = sum(status_trend[:3]) / 3  # Last 3 readings
            if recent_avg < 0.7:  # Below WARNING threshold
                status_degradation_factor = min(0.4, (0.7 - recent_avg) * 2)
        
        # Age factor (older components more likely to fail)
        age_factor = min(0.2, len(data_points) / 100 * 0.1)  # Rough age estimation
        
        # Calculate final failure probability
        failure_probability = min(1.0, base_risk + trend_factor + temp_factor + error_factor + status_degradation_factor + age_factor)
        
        # Estimate time to failure
        if failure_probability > 0.8:
            estimated_ttf_hours = prediction_horizon_hours * 0.1  # Very soon
        elif failure_probability > 0.6:
            estimated_ttf_hours = prediction_horizon_hours * 0.3
        elif failure_probability > 0.4:
            estimated_ttf_hours = prediction_horizon_hours * 0.6
        else:
            estimated_ttf_hours = prediction_horizon_hours  # Beyond prediction horizon
        
        return {
            "component_type": component_type,
            "failure_probability": round(failure_probability, 3),
            "estimated_time_to_failure_hours": round(estimated_ttf_hours, 1),
            "risk_factors": {
                "base_risk": round(base_risk, 3),
                "trend_factor": round(trend_factor, 3),
                "temperature_factor": round(temp_factor, 3),
                "error_factor": round(error_factor, 3),
                "age_factor": round(age_factor, 3)
            },
            "current_status": data_points[0]["status"] if data_points else "UNKNOWN",
            "trend_slope": round(trend_slope, 3),
            "average_temperature": round(sum(temperature_trend) / len(temperature_trend), 1) if temperature_trend else None,
            "error_frequency": round(error_frequency / len(data_points), 2) if data_points else 0
        }
        
    except Exception as e:
        return {
            "component_type": component_type,
            "failure_probability": 0.5,  # Default moderate risk
            "estimated_time_to_failure_hours": prediction_horizon_hours,
            "error": f"Analysis failed: {e}"
        }

def calculate_trend_slope(values: List[float]) -> float:
    """
    Calculate the slope of a trend line using simple linear regression
    
    Args:
        values: List of numeric values in chronological order
        
    Returns:
        Slope value (positive = improving, negative = degrading)
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    x_values = list(range(n))
    
    # Calculate means
    x_mean = sum(x_values) / n
    y_mean = sum(values) / n
    
    # Calculate slope using least squares
    numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def generate_predictive_maintenance_recommendations(predictions: List[Dict]) -> List[Dict]:
    """
    Generate maintenance recommendations based on failure predictions
    
    Args:
        predictions: List of component failure predictions
        
    Returns:
        List of maintenance recommendations
    """
    recommendations = []
    
    for prediction in predictions:
        component_type = prediction["component_type"]
        failure_prob = prediction["failure_probability"]
        ttf_hours = prediction.get("estimated_time_to_failure_hours", 168)
        
        if failure_prob >= 0.8:
            priority = "CRITICAL"
            action = "IMMEDIATE_REPLACEMENT"
            timeframe = "Within 24 hours"
        elif failure_prob >= 0.6:
            priority = "HIGH"
            action = "SCHEDULED_REPLACEMENT"
            timeframe = "Within 72 hours"
        elif failure_prob >= 0.4:
            priority = "MEDIUM"
            action = "PREVENTIVE_MAINTENANCE"
            timeframe = "Within 1 week"
        else:
            priority = "LOW"
            action = "ROUTINE_INSPECTION"
            timeframe = "Within 1 month"
        
        recommendation = {
            "component_type": component_type,
            "priority": priority,
            "recommended_action": action,
            "timeframe": timeframe,
            "failure_probability": failure_prob,
            "estimated_time_to_failure_hours": ttf_hours,
            "description": f"{action.replace('_', ' ').title()} recommended for {component_type} component",
            "cost_impact": estimate_maintenance_cost(component_type, action),
            "downtime_impact": estimate_downtime_impact(component_type, action)
        }
        
        recommendations.append(recommendation)
    
    # Sort by priority and failure probability
    priority_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    recommendations.sort(key=lambda x: (priority_order[x["priority"]], x["failure_probability"]), reverse=True)
    
    return recommendations

def estimate_maintenance_cost(component_type: str, action: str) -> Dict[str, Any]:
    """
    Estimate maintenance cost based on component type and action
    
    Args:
        component_type: Type of component
        action: Maintenance action to be taken
        
    Returns:
        Dictionary containing cost estimates
    """
    # Base cost estimates (in USD)
    component_costs = {
        "POWER": {"parts": 2500, "labor": 800, "downtime": 5000},
        "COOLING": {"parts": 1200, "labor": 600, "downtime": 2000},
        "PROCESSOR": {"parts": 3500, "labor": 1000, "downtime": 8000},
        "ANTENNA": {"parts": 4000, "labor": 1200, "downtime": 10000},
        "MEMORY": {"parts": 800, "labor": 400, "downtime": 1500},
        "NETWORK": {"parts": 1500, "labor": 700, "downtime": 3000}
    }
    
    action_multipliers = {
        "IMMEDIATE_REPLACEMENT": 1.5,  # Emergency costs
        "SCHEDULED_REPLACEMENT": 1.0,
        "PREVENTIVE_MAINTENANCE": 0.3,
        "ROUTINE_INSPECTION": 0.1
    }
    
    base_costs = component_costs.get(component_type, {"parts": 1000, "labor": 500, "downtime": 2000})
    multiplier = action_multipliers.get(action, 1.0)
    
    return {
        "parts_cost": int(base_costs["parts"] * multiplier),
        "labor_cost": int(base_costs["labor"] * multiplier),
        "downtime_cost": int(base_costs["downtime"] * multiplier),
        "total_cost": int((base_costs["parts"] + base_costs["labor"] + base_costs["downtime"]) * multiplier),
        "currency": "USD"
    }

def estimate_downtime_impact(component_type: str, action: str) -> Dict[str, Any]:
    """
    Estimate downtime impact based on component type and maintenance action
    
    Args:
        component_type: Type of component
        action: Maintenance action to be taken
        
    Returns:
        Dictionary containing downtime estimates
    """
    # Base downtime estimates (in hours)
    component_downtime = {
        "POWER": {"replacement": 4, "maintenance": 1, "inspection": 0.5},
        "COOLING": {"replacement": 2, "maintenance": 0.5, "inspection": 0.25},
        "PROCESSOR": {"replacement": 6, "maintenance": 2, "inspection": 0.5},
        "ANTENNA": {"replacement": 8, "maintenance": 3, "inspection": 1},
        "MEMORY": {"replacement": 1, "maintenance": 0.5, "inspection": 0.25},
        "NETWORK": {"replacement": 3, "maintenance": 1, "inspection": 0.5}
    }
    
    action_mapping = {
        "IMMEDIATE_REPLACEMENT": "replacement",
        "SCHEDULED_REPLACEMENT": "replacement",
        "PREVENTIVE_MAINTENANCE": "maintenance",
        "ROUTINE_INSPECTION": "inspection"
    }
    
    downtime_type = action_mapping.get(action, "maintenance")
    base_downtime = component_downtime.get(component_type, {"replacement": 3, "maintenance": 1, "inspection": 0.5})
    
    estimated_hours = base_downtime[downtime_type]
    
    # Emergency work takes longer
    if action == "IMMEDIATE_REPLACEMENT":
        estimated_hours *= 1.5
    
    return {
        "estimated_downtime_hours": estimated_hours,
        "service_impact": "FULL" if estimated_hours > 2 else "PARTIAL" if estimated_hours > 0.5 else "MINIMAL",
        "affected_users": estimate_affected_users(component_type, estimated_hours),
        "revenue_impact": estimate_revenue_impact(estimated_hours)
    }

def estimate_affected_users(component_type: str, downtime_hours: float) -> int:
    """
    Estimate number of users affected by component downtime
    
    Args:
        component_type: Type of component
        downtime_hours: Expected downtime in hours
        
    Returns:
        Estimated number of affected users
    """
    # Base users per tower (typical 5G coverage)
    base_users = 5000
    
    # Component impact multipliers
    impact_multipliers = {
        "POWER": 1.0,      # Full tower down
        "PROCESSOR": 1.0,  # Full tower down
        "ANTENNA": 0.8,    # Reduced coverage
        "COOLING": 0.6,    # Gradual degradation
        "NETWORK": 0.7,    # Connectivity issues
        "MEMORY": 0.4      # Performance degradation
    }
    
    multiplier = impact_multipliers.get(component_type, 0.5)
    
    # Time-of-day factor (more users during peak hours)
    time_factor = 0.7  # Average factor
    
    return int(base_users * multiplier * time_factor)

def estimate_revenue_impact(downtime_hours: float) -> Dict[str, Any]:
    """
    Estimate revenue impact of downtime
    
    Args:
        downtime_hours: Expected downtime in hours
        
    Returns:
        Dictionary containing revenue impact estimates
    """
    # Average revenue per user per hour (estimated)
    revenue_per_user_hour = 0.15  # $0.15 per user per hour
    
    # Base users affected (will be calculated per component)
    base_users = 3000  # Conservative estimate
    
    direct_revenue_loss = base_users * revenue_per_user_hour * downtime_hours
    
    # SLA penalties (estimated)
    sla_penalty = direct_revenue_loss * 0.5
    
    # Customer churn impact (long-term)
    churn_impact = direct_revenue_loss * 2.0 if downtime_hours > 4 else 0
    
    return {
        "direct_revenue_loss": round(direct_revenue_loss, 2),
        "sla_penalties": round(sla_penalty, 2),
        "churn_impact": round(churn_impact, 2),
        "total_impact": round(direct_revenue_loss + sla_penalty + churn_impact, 2),
        "currency": "USD"
    }

def schedule_proactive_maintenance(tower_id: str, component_type: str, 
                                 priority: str, recommended_date: str = None) -> Dict[str, Any]:
    """
    Schedule proactive maintenance based on predictive analysis
    
    Args:
        tower_id: The ID of the tower
        component_type: Type of component requiring maintenance
        priority: Priority level (LOW, MEDIUM, HIGH, CRITICAL)
        recommended_date: Optional recommended maintenance date
        
    Returns:
        Dictionary containing scheduling result
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Generate maintenance description based on prediction
        description = f"Proactive maintenance scheduled for {component_type} component based on predictive analysis"
        
        # Create maintenance ticket
        ticket_result = create_maintenance_ticket(
            tower_id=tower_id,
            issue_description=description,
            priority=priority,
            assigned_to=None  # Will be auto-assigned based on priority
        )
        
        if not ticket_result["success"]:
            conn.close()
            return ticket_result
        
        ticket_id = ticket_result["ticket_id"]
        
        # Create maintenance schedule entry
        cursor.execute("""
            INSERT INTO maintenance_schedules 
            (ticket_id, tower_id, component_type, scheduled_date, maintenance_type, priority, status)
            VALUES (?, ?, ?, ?, 'PROACTIVE', ?, 'SCHEDULED')
        """, (
            ticket_id, tower_id, component_type, 
            recommended_date or (datetime.now() + timedelta(days=7)).isoformat(),
            priority
        ))
        
        schedule_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "message": f"Proactive maintenance scheduled for {tower_id} {component_type}",
            "schedule_id": schedule_id,
            "ticket_id": ticket_id,
            "tower_id": tower_id,
            "component_type": component_type,
            "priority": priority,
            "scheduled_date": recommended_date or (datetime.now() + timedelta(days=7)).isoformat(),
            "maintenance_type": "PROACTIVE"
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error scheduling maintenance: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error scheduling maintenance: {e}"
        }

def get_equipment_lifecycle_status(tower_id: str = None, component_type: str = None) -> Dict[str, Any]:
    """
    Get equipment lifecycle status and recommendations
    
    Args:
        tower_id: Optional filter by tower ID
        component_type: Optional filter by component type
        
    Returns:
        Dictionary containing lifecycle status information
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query for equipment lifecycle analysis
        base_query = """
            SELECT h.tower_id, h.component_type, 
                   MIN(h.last_checked) as first_seen,
                   MAX(h.last_checked) as last_seen,
                   COUNT(*) as status_changes,
                   AVG(CASE 
                       WHEN h.status = 'HEALTHY' THEN 100
                       WHEN h.status = 'WARNING' THEN 70
                       WHEN h.status = 'CRITICAL' THEN 30
                       WHEN h.status = 'FAILED' THEN 0
                       ELSE 50
                   END) as avg_health_score,
                   t.name as tower_name
            FROM hardware_status h
            LEFT JOIN towers t ON h.tower_id = t.id
            WHERE 1=1
        """
        params = []
        
        if tower_id:
            base_query += " AND h.tower_id = ?"
            params.append(tower_id)
        
        if component_type:
            base_query += " AND h.component_type = ?"
            params.append(component_type)
        
        base_query += """
            GROUP BY h.tower_id, h.component_type
            ORDER BY h.tower_id, h.component_type
        """
        
        cursor.execute(base_query, params)
        results = cursor.fetchall()
        
        lifecycle_data = []
        
        for row in results:
            # Calculate equipment age and lifecycle stage
            first_seen = datetime.fromisoformat(row["first_seen"])
            last_seen = datetime.fromisoformat(row["last_seen"])
            age_days = (datetime.now() - first_seen).days
            
            # Determine lifecycle stage based on age and health
            lifecycle_stage = determine_lifecycle_stage(
                age_days, row["avg_health_score"], row["status_changes"]
            )
            
            # Calculate replacement recommendation
            replacement_recommendation = calculate_replacement_timeline(
                row["component_type"], age_days, row["avg_health_score"]
            )
            
            lifecycle_entry = {
                "tower_id": row["tower_id"],
                "tower_name": row["tower_name"] or "Unknown",
                "component_type": row["component_type"],
                "age_days": age_days,
                "lifecycle_stage": lifecycle_stage,
                "average_health_score": round(row["avg_health_score"], 1),
                "status_changes": row["status_changes"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "replacement_recommendation": replacement_recommendation
            }
            
            lifecycle_data.append(lifecycle_entry)
        
        # Generate summary statistics
        if lifecycle_data:
            avg_age = sum(item["age_days"] for item in lifecycle_data) / len(lifecycle_data)
            avg_health = sum(item["average_health_score"] for item in lifecycle_data) / len(lifecycle_data)
            
            stage_counts = {}
            for item in lifecycle_data:
                stage = item["lifecycle_stage"]
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        else:
            avg_age = 0
            avg_health = 0
            stage_counts = {}
        
        conn.close()
        
        return {
            "success": True,
            "message": f"Equipment lifecycle status retrieved for {len(lifecycle_data)} components",
            "total_components": len(lifecycle_data),
            "average_age_days": round(avg_age, 1),
            "average_health_score": round(avg_health, 1),
            "lifecycle_stage_distribution": stage_counts,
            "equipment_details": lifecycle_data,
            "filters_applied": {
                "tower_id": tower_id,
                "component_type": component_type
            }
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error getting lifecycle status: {e}",
            "equipment_details": []
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error getting lifecycle status: {e}",
            "equipment_details": []
        }

def determine_lifecycle_stage(age_days: int, avg_health_score: float, status_changes: int) -> str:
    """
    Determine equipment lifecycle stage based on age, health, and stability
    
    Args:
        age_days: Age of equipment in days
        avg_health_score: Average health score (0-100)
        status_changes: Number of status changes (indicator of stability)
        
    Returns:
        Lifecycle stage string
    """
    # Define lifecycle stages based on typical telecom equipment lifespan
    if age_days < 90:  # 3 months
        if avg_health_score >= 90:
            return "NEW"
        else:
            return "BURN_IN"  # Early issues during burn-in period
    elif age_days < 1095:  # 3 years
        if avg_health_score >= 80 and status_changes < 10:
            return "STABLE"
        elif avg_health_score >= 60:
            return "OPERATIONAL"
        else:
            return "PROBLEMATIC"
    elif age_days < 1825:  # 5 years
        if avg_health_score >= 70:
            return "MATURE"
        else:
            return "AGING"
    else:  # > 5 years
        if avg_health_score >= 60:
            return "LEGACY"
        else:
            return "END_OF_LIFE"

def calculate_replacement_timeline(component_type: str, age_days: int, avg_health_score: float) -> Dict[str, Any]:
    """
    Calculate recommended replacement timeline for equipment
    
    Args:
        component_type: Type of component
        age_days: Current age in days
        avg_health_score: Average health score
        
    Returns:
        Dictionary containing replacement recommendations
    """
    # Typical lifespan for telecom components (in days)
    component_lifespans = {
        "POWER": 2555,      # 7 years
        "COOLING": 1825,    # 5 years
        "PROCESSOR": 2190,  # 6 years
        "ANTENNA": 3650,    # 10 years
        "MEMORY": 2190,     # 6 years
        "NETWORK": 2555     # 7 years
    }
    
    expected_lifespan = component_lifespans.get(component_type, 2190)  # Default 6 years
    remaining_lifespan = max(0, expected_lifespan - age_days)
    
    # Adjust based on health score
    health_factor = avg_health_score / 100
    adjusted_remaining = int(remaining_lifespan * health_factor)
    
    # More aggressive urgency calculation for poor health
    if avg_health_score < 40 or adjusted_remaining < 30:  # Very poor health or less than 1 month
        urgency = "CRITICAL"
        timeframe = "Within 30 days"
    elif avg_health_score < 60 or adjusted_remaining < 90:  # Poor health or less than 3 months
        urgency = "HIGH"
        timeframe = "Within 90 days"
    elif avg_health_score < 80 or adjusted_remaining < 365:  # Fair health or less than 1 year
        urgency = "MEDIUM"
        timeframe = "Within 1 year"
    else:
        urgency = "LOW"
        timeframe = f"In {max(1, adjusted_remaining // 365)} years"
    
    return {
        "urgency": urgency,
        "timeframe": timeframe,
        "remaining_lifespan_days": adjusted_remaining,
        "expected_total_lifespan_days": expected_lifespan,
        "age_percentage": round((age_days / expected_lifespan) * 100, 1),
        "health_adjusted": health_factor < 0.8,  # Health significantly impacts lifespan
        "recommendation": f"Plan replacement {timeframe.lower()} based on {urgency.lower()} priority"
    }

def analyze_error_codes(error_codes: str) -> int:
    """
    Analyze error codes to determine severity impact on priority
    
    Args:
        error_codes: Comma-separated error codes
        
    Returns:
        Priority score adjustment (0-25)
    """
    if not error_codes:
        return 0
    
    # Critical error patterns
    critical_patterns = ["CRIT_", "FATAL_", "EMERG_", "FAIL_"]
    high_patterns = ["ERR_", "ERROR_", "WARN_"]
    
    codes = error_codes.upper().split(",")
    max_severity = 0
    
    for code in codes:
        code = code.strip()
        if any(pattern in code for pattern in critical_patterns):
            max_severity = max(max_severity, 25)
        elif any(pattern in code for pattern in high_patterns):
            max_severity = max(max_severity, 15)
        else:
            max_severity = max(max_severity, 5)
    
    return max_severity

def analyze_performance_impact(metrics: Dict[str, Any]) -> int:
    """
    Analyze performance metrics to determine impact on priority
    
    Args:
        metrics: Performance metrics dictionary
        
    Returns:
        Priority score adjustment (0-20)
    """
    impact_score = 0
    
    # CPU usage impact
    cpu_usage = metrics.get("cpu_usage", 0)
    if cpu_usage > 95:
        impact_score += 15
    elif cpu_usage > 85:
        impact_score += 10
    elif cpu_usage > 75:
        impact_score += 5
    
    # Memory usage impact
    memory_usage = metrics.get("memory_usage", 0)
    if memory_usage > 90:
        impact_score += 10
    elif memory_usage > 80:
        impact_score += 5
    
    # Signal strength impact
    signal_strength = metrics.get("signal_strength", -70)
    if signal_strength < -90:
        impact_score += 15
    elif signal_strength < -80:
        impact_score += 10
    
    # Error rate impact
    error_rate = metrics.get("error_rate", 0)
    if error_rate > 5:
        impact_score += 10
    elif error_rate > 2:
        impact_score += 5
    
    return min(impact_score, 20)

def get_priority_recommendation(priority: str, component_type: str) -> str:
    """
    Get recommendation based on priority and component type
    
    Args:
        priority: Priority level
        component_type: Type of component
        
    Returns:
        Recommendation string
    """
    recommendations = {
        "CRITICAL": {
            "POWER": "Immediate dispatch required - Tower may go offline",
            "COOLING": "Emergency response needed - Risk of equipment damage",
            "PROCESSOR": "Critical system failure - Service interruption likely",
            "default": "Immediate attention required - High service impact"
        },
        "HIGH": {
            "POWER": "Urgent maintenance needed within 2 hours",
            "COOLING": "Schedule emergency maintenance within 4 hours",
            "ANTENNA": "Signal quality degraded - Schedule repair within 8 hours",
            "default": "High priority maintenance needed within 8 hours"
        },
        "MEDIUM": {
            "default": "Schedule maintenance within 24-48 hours"
        },
        "LOW": {
            "default": "Include in next scheduled maintenance window"
        }
    }
    
    priority_recs = recommendations.get(priority, recommendations["MEDIUM"])
    return priority_recs.get(component_type, priority_recs["default"])

def notify_field_team(ticket_id: int, notification_type: str = "ASSIGNMENT", 
                     urgency_level: str = "NORMAL") -> Dict[str, Any]:
    """
    Enhanced field team notification system with multiple channels and escalation
    
    Args:
        ticket_id: ID of the maintenance ticket
        notification_type: Type of notification (ASSIGNMENT, ESCALATION, COMPLETION, UPDATE)
        urgency_level: Urgency level (LOW, NORMAL, HIGH, CRITICAL)
        
    Returns:
        Dictionary containing notification result
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get comprehensive ticket details
        cursor.execute("""
            SELECT mt.*, t.name as tower_name, t.latitude, t.longitude, t.status as tower_status,
                   hs.status as hardware_status, hs.component_type, hs.error_codes
            FROM maintenance_tickets mt
            LEFT JOIN towers t ON mt.tower_id = t.id
            LEFT JOIN hardware_status hs ON mt.tower_id = hs.tower_id
            WHERE mt.id = ?
            ORDER BY hs.last_checked DESC
            LIMIT 1
        """, (ticket_id,))
        
        ticket = cursor.fetchone()
        conn.close()
        
        if not ticket:
            return {
                "success": False,
                "message": f"Maintenance ticket {ticket_id} not found"
            }
        
        # Determine notification channels based on priority and urgency
        notification_channels = determine_notification_channels(
            ticket["priority"], urgency_level, notification_type
        )
        
        # Generate notification content
        notification_content = generate_notification_content(ticket, notification_type)
        
        # Simulate multi-channel notification delivery
        delivery_results = []
        
        for channel in notification_channels:
            result = simulate_notification_delivery(
                channel, ticket, notification_content, urgency_level
            )
            delivery_results.append(result)
        
        # Log notification for audit trail
        log_notification_event(ticket_id, notification_type, notification_channels, urgency_level)
        
        # Determine if escalation is needed
        escalation_needed = should_escalate_notification(
            ticket["priority"], notification_type, urgency_level
        )
        
        return {
            "success": True,
            "message": f"Field team notified about ticket {ticket_id} via {len(notification_channels)} channels",
            "ticket_id": ticket_id,
            "notification_type": notification_type,
            "urgency_level": urgency_level,
            "channels_used": notification_channels,
            "delivery_results": delivery_results,
            "escalation_needed": escalation_needed,
            "notification_sent_to": ticket["assigned_to"] or "All available technicians",
            "tower_details": {
                "tower_id": ticket["tower_id"],
                "tower_name": ticket["tower_name"],
                "location": f"{ticket['latitude']}, {ticket['longitude']}",
                "tower_status": ticket["tower_status"],
                "hardware_status": ticket["hardware_status"]
            },
            "estimated_response_time": estimate_response_time(ticket["priority"], urgency_level)
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error sending notification: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error sending notification: {e}"
        }

def determine_notification_channels(priority: str, urgency_level: str, notification_type: str) -> List[str]:
    """
    Determine which notification channels to use based on priority and urgency
    
    Args:
        priority: Ticket priority (LOW, MEDIUM, HIGH, CRITICAL)
        urgency_level: Urgency level (LOW, NORMAL, HIGH, CRITICAL)
        notification_type: Type of notification
        
    Returns:
        List of notification channels to use
    """
    channels = ["dashboard"]  # Always include dashboard
    
    if priority in ["HIGH", "CRITICAL"] or urgency_level in ["HIGH", "CRITICAL"]:
        channels.extend(["sms", "email", "mobile_app"])
        
        if priority == "CRITICAL" or urgency_level == "CRITICAL":
            channels.append("phone_call")
    elif priority == "MEDIUM" or urgency_level == "NORMAL":
        channels.extend(["email", "mobile_app"])
    else:  # LOW priority
        channels.append("email")
    
    # Special handling for escalations
    if notification_type == "ESCALATION":
        channels.extend(["sms", "phone_call"])
        if "supervisor_alert" not in channels:
            channels.append("supervisor_alert")
    
    return list(set(channels))  # Remove duplicates

def generate_notification_content(ticket: Dict[str, Any], notification_type: str) -> Dict[str, str]:
    """
    Generate notification content for different channels
    
    Args:
        ticket: Ticket information
        notification_type: Type of notification
        
    Returns:
        Dictionary with content for different channels
    """
    # Base information
    tower_info = f"{ticket['tower_name']} ({ticket['tower_id']})"
    location = f"{ticket['latitude']}, {ticket['longitude']}"
    
    # Short message for SMS/mobile
    short_message = f"MAINT {notification_type}: {ticket['priority']} priority issue at {tower_info}. Ticket #{ticket['id']}"
    
    # Detailed message for email/dashboard
    detailed_message = f"""
MAINTENANCE NOTIFICATION - {notification_type}

Ticket Details:
- ID: {ticket['id']}
- Priority: {ticket['priority']}
- Status: {ticket['status']}
- Tower: {tower_info}
- Location: {location}
- Assigned To: {ticket['assigned_to'] or 'Unassigned'}

Issue Description:
{ticket['issue_description']}

Hardware Status: {ticket['hardware_status'] if ticket['hardware_status'] else 'Unknown'}
Component: {ticket['component_type'] if ticket['component_type'] else 'Unknown'}
Error Codes: {ticket['error_codes'] if ticket['error_codes'] else 'None'}

Created: {ticket['created_at']}
{f"Resolved: {ticket['resolved_at']}" if ticket['resolved_at'] else ""}

Action Required: {get_action_required(notification_type, ticket['priority'])}
"""
    
    return {
        "short": short_message,
        "detailed": detailed_message,
        "subject": f"Maintenance {notification_type} - {ticket['priority']} Priority - {tower_info}"
    }

def simulate_notification_delivery(channel: str, ticket: Dict[str, Any], 
                                 content: Dict[str, str], urgency_level: str) -> Dict[str, Any]:
    """
    Simulate notification delivery to a specific channel
    
    Args:
        channel: Notification channel
        ticket: Ticket information
        content: Notification content
        urgency_level: Urgency level
        
    Returns:
        Delivery result dictionary
    """
    # Simulate delivery with realistic success rates and timing
    import random
    import time
    
    delivery_times = {
        "dashboard": 1,
        "email": 5,
        "sms": 10,
        "mobile_app": 3,
        "phone_call": 30,
        "supervisor_alert": 15
    }
    
    success_rates = {
        "dashboard": 0.99,
        "email": 0.95,
        "sms": 0.98,
        "mobile_app": 0.92,
        "phone_call": 0.85,
        "supervisor_alert": 0.97
    }
    
    delivery_time = delivery_times.get(channel, 5)
    success_rate = success_rates.get(channel, 0.9)
    
    # Higher urgency improves success rates
    if urgency_level in ["HIGH", "CRITICAL"]:
        success_rate = min(success_rate + 0.05, 1.0)
    
    success = random.random() < success_rate
    
    return {
        "channel": channel,
        "success": success,
        "delivery_time_seconds": delivery_time,
        "message_sent": content["short"] if channel in ["sms", "mobile_app"] else content["detailed"],
        "recipient": ticket["assigned_to"] or "On-call team",
        "timestamp": datetime.now().isoformat()
    }

def log_notification_event(ticket_id: int, notification_type: str, 
                          channels: List[str], urgency_level: str):
    """
    Log notification event for audit trail
    
    Args:
        ticket_id: Ticket ID
        notification_type: Type of notification
        channels: Channels used
        urgency_level: Urgency level
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Log to network_events table
        # Map urgency levels to valid severity values
        severity_mapping = {
            "LOW": "INFO",
            "NORMAL": "INFO", 
            "HIGH": "WARNING",
            "CRITICAL": "CRITICAL"
        }
        severity = severity_mapping.get(urgency_level, "INFO")
        
        cursor.execute("""
            INSERT INTO network_events (event_type, timestamp, severity, description, metadata)
            VALUES ('MAINTENANCE', ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            severity,
            f"Notification sent for maintenance ticket {ticket_id}",
            json.dumps({
                "ticket_id": ticket_id,
                "notification_type": notification_type,
                "channels": channels,
                "urgency_level": urgency_level
            })
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logging.error(f"Failed to log notification event: {e}")

def should_escalate_notification(priority: str, notification_type: str, urgency_level: str) -> bool:
    """
    Determine if notification should be escalated
    
    Args:
        priority: Ticket priority
        notification_type: Type of notification
        urgency_level: Urgency level
        
    Returns:
        True if escalation is needed
    """
    # Always escalate critical issues
    if priority == "CRITICAL" or urgency_level == "CRITICAL":
        return True
    
    # Escalate high priority issues that aren't assignments
    if priority == "HIGH" and notification_type != "ASSIGNMENT":
        return True
    
    # Escalate if this is already an escalation
    if notification_type == "ESCALATION":
        return True
    
    return False

def estimate_response_time(priority: str, urgency_level: str) -> str:
    """
    Estimate response time based on priority and urgency
    
    Args:
        priority: Ticket priority
        urgency_level: Urgency level
        
    Returns:
        Estimated response time string
    """
    if priority == "CRITICAL" or urgency_level == "CRITICAL":
        return "15-30 minutes"
    elif priority == "HIGH" or urgency_level == "HIGH":
        return "1-2 hours"
    elif priority == "MEDIUM":
        return "4-8 hours"
    else:
        return "24-48 hours"

def get_action_required(notification_type: str, priority: str) -> str:
    """
    Get action required message based on notification type and priority
    
    Args:
        notification_type: Type of notification
        priority: Ticket priority
        
    Returns:
        Action required message
    """
    actions = {
        "ASSIGNMENT": {
            "CRITICAL": "Immediate dispatch required - Drop current tasks",
            "HIGH": "Urgent response needed within 2 hours",
            "MEDIUM": "Schedule response within 8 hours",
            "LOW": "Include in next maintenance round"
        },
        "ESCALATION": {
            "CRITICAL": "Supervisor intervention required immediately",
            "HIGH": "Additional resources needed urgently",
            "MEDIUM": "Review and reassign if necessary",
            "LOW": "Monitor progress and update timeline"
        },
        "UPDATE": {
            "default": "Review updated information and adjust response accordingly"
        },
        "COMPLETION": {
            "default": "Verify resolution and close ticket"
        }
    }
    
    type_actions = actions.get(notification_type, actions["UPDATE"])
    return type_actions.get(priority, type_actions.get("default", "Review and take appropriate action"))

def auto_assign_ticket(ticket_id: int, assignment_criteria: str = "workload") -> Dict[str, Any]:
    """
    Automatically assign a maintenance ticket to an available technician
    
    Args:
        ticket_id: ID of the ticket to assign
        assignment_criteria: Criteria for assignment (workload, location, expertise)
        
    Returns:
        Dictionary containing assignment result
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get ticket details
        cursor.execute("""
            SELECT mt.*, t.latitude, t.longitude, t.name as tower_name
            FROM maintenance_tickets mt
            LEFT JOIN towers t ON mt.tower_id = t.id
            WHERE mt.id = ? AND mt.status = 'OPEN'
        """, (ticket_id,))
        
        ticket = cursor.fetchone()
        if not ticket:
            conn.close()
            return {
                "success": False,
                "message": f"Open ticket {ticket_id} not found"
            }
        
        # Simulate technician selection based on criteria
        available_technicians = get_available_technicians(
            ticket["priority"], ticket["latitude"], ticket["longitude"], assignment_criteria
        )
        
        if not available_technicians:
            conn.close()
            return {
                "success": False,
                "message": "No available technicians found for assignment"
            }
        
        # Select best technician
        selected_tech = available_technicians[0]
        
        # Update ticket assignment
        cursor.execute("""
            UPDATE maintenance_tickets 
            SET assigned_to = ?, status = 'ASSIGNED'
            WHERE id = ?
        """, (selected_tech["id"], ticket_id))
        
        conn.commit()
        conn.close()
        
        # Send notification
        notify_result = notify_field_team(ticket_id, "ASSIGNMENT", "NORMAL")
        
        return {
            "success": True,
            "message": f"Ticket {ticket_id} assigned to {selected_tech['name']}",
            "ticket_id": ticket_id,
            "assigned_to": selected_tech["id"],
            "technician_name": selected_tech["name"],
            "assignment_criteria": assignment_criteria,
            "estimated_arrival": selected_tech["estimated_arrival"],
            "notification_sent": notify_result["success"]
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error during assignment: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error during assignment: {e}"
        }

def get_available_technicians(priority: str, tower_lat: float, tower_lon: float, 
                            criteria: str) -> List[Dict[str, Any]]:
    """
    Get list of available technicians based on assignment criteria
    
    Args:
        priority: Ticket priority
        tower_lat: Tower latitude
        tower_lon: Tower longitude
        criteria: Assignment criteria
        
    Returns:
        List of available technicians sorted by suitability
    """
    # Simulate technician database (in real implementation, this would be a proper database)
    technicians = [
        {
            "id": "TECH_001",
            "name": "John Smith",
            "location": (40.7128, -74.0060),
            "specialties": ["ANTENNA", "NETWORK"],
            "current_workload": 2,
            "max_workload": 5,
            "shift_end": "18:00"
        },
        {
            "id": "TECH_002", 
            "name": "Sarah Johnson",
            "location": (40.7589, -73.9851),
            "specialties": ["POWER", "COOLING"],
            "current_workload": 1,
            "max_workload": 4,
            "shift_end": "17:00"
        },
        {
            "id": "TECH_003",
            "name": "Mike Chen",
            "location": (40.7831, -73.9712),
            "specialties": ["PROCESSOR", "MEMORY"],
            "current_workload": 3,
            "max_workload": 5,
            "shift_end": "19:00"
        }
    ]
    
    # Filter available technicians
    available = []
    for tech in technicians:
        if tech["current_workload"] < tech["max_workload"]:
            # Calculate distance (simplified)
            distance = abs(tech["location"][0] - tower_lat) + abs(tech["location"][1] - tower_lon)
            
            # Calculate travel time (simplified: 1 degree ≈ 30 minutes)
            travel_time = int(distance * 30)
            
            tech_info = {
                **tech,
                "distance": round(distance, 4),
                "travel_time_minutes": travel_time,
                "estimated_arrival": f"{travel_time} minutes"
            }
            available.append(tech_info)
    
    # Sort by criteria
    if criteria == "workload":
        available.sort(key=lambda x: x["current_workload"])
    elif criteria == "location":
        available.sort(key=lambda x: x["distance"])
    elif criteria == "expertise":
        # Priority-based sorting would require component type info
        available.sort(key=lambda x: (x["current_workload"], x["distance"]))
    
    return available

def escalate_ticket(ticket_id: int, escalation_reason: str, escalate_to: str = "supervisor") -> Dict[str, Any]:
    """
    Escalate a maintenance ticket to higher priority or different team
    
    Args:
        ticket_id: ID of the ticket to escalate
        escalation_reason: Reason for escalation
        escalate_to: Who to escalate to (supervisor, specialist, emergency)
        
    Returns:
        Dictionary containing escalation result
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current ticket
        cursor.execute("SELECT * FROM maintenance_tickets WHERE id = ?", (ticket_id,))
        ticket = cursor.fetchone()
        
        if not ticket:
            conn.close()
            return {
                "success": False,
                "message": f"Ticket {ticket_id} not found"
            }
        
        # Determine new priority and assignment
        current_priority = ticket["priority"]
        new_priority = escalate_priority(current_priority)
        new_assignee = determine_escalation_assignee(escalate_to, new_priority)
        
        # Update ticket
        cursor.execute("""
            UPDATE maintenance_tickets 
            SET priority = ?, assigned_to = ?, status = 'ASSIGNED',
                resolution_notes = COALESCE(resolution_notes || '\n', '') || ?
            WHERE id = ?
        """, (
            new_priority, 
            new_assignee,
            f"ESCALATED: {escalation_reason} (from {current_priority} to {new_priority})",
            ticket_id
        ))
        
        conn.commit()
        conn.close()
        
        # Send escalation notification
        notify_result = notify_field_team(ticket_id, "ESCALATION", "HIGH")
        
        return {
            "success": True,
            "message": f"Ticket {ticket_id} escalated from {current_priority} to {new_priority}",
            "ticket_id": ticket_id,
            "previous_priority": current_priority,
            "new_priority": new_priority,
            "escalated_to": new_assignee,
            "escalation_reason": escalation_reason,
            "notification_sent": notify_result["success"]
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error during escalation: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error during escalation: {e}"
        }

def escalate_priority(current_priority: str) -> str:
    """Escalate priority to next level"""
    escalation_map = {
        "LOW": "MEDIUM",
        "MEDIUM": "HIGH", 
        "HIGH": "CRITICAL",
        "CRITICAL": "CRITICAL"  # Already at max
    }
    return escalation_map.get(current_priority, "HIGH")

def determine_escalation_assignee(escalate_to: str, priority: str) -> str:
    """Determine who to assign escalated ticket to"""
    if escalate_to == "supervisor":
        return "SUPERVISOR_001"
    elif escalate_to == "specialist":
        return "SPECIALIST_001"
    elif escalate_to == "emergency":
        return "EMERGENCY_TEAM"
    else:
        return "ESCALATION_TEAM"

def get_ticket_metrics(time_period_hours: int = 24) -> Dict[str, Any]:
    """
    Get maintenance ticket metrics and analytics
    
    Args:
        time_period_hours: Time period for metrics calculation
        
    Returns:
        Dictionary containing ticket metrics
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate time threshold
        threshold_time = datetime.now() - timedelta(hours=time_period_hours)
        
        # Get ticket counts by status
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM maintenance_tickets
            WHERE created_at >= ?
            GROUP BY status
        """, (threshold_time.isoformat(),))
        
        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        # Get ticket counts by priority
        cursor.execute("""
            SELECT priority, COUNT(*) as count
            FROM maintenance_tickets
            WHERE created_at >= ?
            GROUP BY priority
        """, (threshold_time.isoformat(),))
        
        priority_counts = {row["priority"]: row["count"] for row in cursor.fetchall()}
        
        # Get average resolution time
        cursor.execute("""
            SELECT AVG(
                CASE 
                    WHEN resolved_at IS NOT NULL 
                    THEN (julianday(resolved_at) - julianday(created_at)) * 24 
                    ELSE NULL 
                END
            ) as avg_resolution_hours
            FROM maintenance_tickets
            WHERE created_at >= ? AND resolved_at IS NOT NULL
        """, (threshold_time.isoformat(),))
        
        avg_resolution = cursor.fetchone()["avg_resolution_hours"] or 0
        
        # Get tickets by tower
        cursor.execute("""
            SELECT tower_id, COUNT(*) as count
            FROM maintenance_tickets
            WHERE created_at >= ?
            GROUP BY tower_id
            ORDER BY count DESC
            LIMIT 5
        """, (threshold_time.isoformat(),))
        
        tower_counts = [{"tower_id": row["tower_id"], "count": row["count"]} 
                       for row in cursor.fetchall()]
        
        # Get overdue tickets
        cursor.execute("""
            SELECT COUNT(*) as overdue_count
            FROM maintenance_tickets
            WHERE status NOT IN ('RESOLVED', 'CLOSED')
            AND created_at < ?
            AND priority IN ('HIGH', 'CRITICAL')
        """, ((datetime.now() - timedelta(hours=4)).isoformat(),))
        
        overdue_count = cursor.fetchone()["overdue_count"]
        
        conn.close()
        
        # Calculate metrics
        total_tickets = sum(status_counts.values())
        resolution_rate = (status_counts.get("RESOLVED", 0) + status_counts.get("CLOSED", 0)) / max(total_tickets, 1) * 100
        
        return {
            "success": True,
            "message": f"Ticket metrics for last {time_period_hours} hours",
            "time_period_hours": time_period_hours,
            "total_tickets": total_tickets,
            "status_breakdown": status_counts,
            "priority_breakdown": priority_counts,
            "resolution_rate_percent": round(resolution_rate, 2),
            "average_resolution_hours": round(avg_resolution, 2),
            "overdue_tickets": overdue_count,
            "top_affected_towers": tower_counts,
            "performance_indicators": {
                "sla_compliance": "GOOD" if avg_resolution < 8 else "NEEDS_IMPROVEMENT",
                "workload_status": "HIGH" if total_tickets > 20 else "NORMAL",
                "escalation_rate": round((priority_counts.get("CRITICAL", 0) / max(total_tickets, 1)) * 100, 2)
            }
        }
        
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"Database error getting metrics: {e}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error getting metrics: {e}"
        }

# --- MCP Server Setup ---
logging.info("Creating Maintenance MCP Server instance...")
app = Server("maintenance-mcp-server")

@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    logging.info("Maintenance MCP Server: Received list_tools request.")
    
    tools = [
        mcp_types.Tool(
            name="monitor_hardware_health",
            description="Monitor hardware health for a specific tower",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "The ID of the tower to monitor"
                    }
                },
                "required": ["tower_id"]
            }
        ),
        mcp_types.Tool(
            name="detect_hardware_anomalies",
            description="Detect hardware anomalies across all towers within the specified time window",
            inputSchema={
                "type": "object",
                "properties": {
                    "threshold_hours": {
                        "type": "integer",
                        "description": "Hours to look back for anomaly detection (default: 24)",
                        "default": 24
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="update_hardware_status",
            description="Update hardware status for a specific tower component",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "The ID of the tower"
                    },
                    "component_type": {
                        "type": "string",
                        "description": "Type of component (ANTENNA, PROCESSOR, MEMORY, POWER, COOLING, NETWORK)"
                    },
                    "status": {
                        "type": "string",
                        "description": "Health status (HEALTHY, WARNING, CRITICAL, FAILED)"
                    },
                    "error_codes": {
                        "type": "string",
                        "description": "Optional error codes",
                        "default": ""
                    },
                    "performance_metrics": {
                        "type": "string",
                        "description": "Optional performance metrics JSON",
                        "default": ""
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Optional temperature reading"
                    }
                },
                "required": ["tower_id", "component_type", "status"]
            }
        ),
        mcp_types.Tool(
            name="get_equipment_status_summary",
            description="Get a summary of equipment status across all towers",
            inputSchema={
                "type": "object",
                "properties": {
                    "dummy_param": {
                        "type": "string",
                        "description": "Unused parameter for schema generation",
                        "default": ""
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="create_maintenance_ticket",
            description="Create a new maintenance ticket for a tower issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "The ID of the tower with the issue"
                    },
                    "issue_description": {
                        "type": "string",
                        "description": "Description of the maintenance issue"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority level (LOW, MEDIUM, HIGH, CRITICAL)",
                        "default": "MEDIUM"
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": "Optional technician assignment"
                    }
                },
                "required": ["tower_id", "issue_description"]
            }
        ),
        mcp_types.Tool(
            name="get_maintenance_tickets",
            description="Retrieve maintenance tickets with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "Optional filter by tower ID"
                    },
                    "status": {
                        "type": "string",
                        "description": "Optional filter by ticket status"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Optional filter by priority level"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tickets to return",
                        "default": 50
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="update_maintenance_ticket",
            description="Update an existing maintenance ticket",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "ID of the ticket to update"
                    },
                    "status": {
                        "type": "string",
                        "description": "New status (OPEN, ASSIGNED, IN_PROGRESS, RESOLVED, CLOSED)"
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": "Technician to assign the ticket to"
                    },
                    "resolution_notes": {
                        "type": "string",
                        "description": "Notes about the resolution"
                    }
                },
                "required": ["ticket_id"]
            }
        ),
        mcp_types.Tool(
            name="classify_issue_priority",
            description="Enhanced priority classification algorithm for maintenance issues",
            inputSchema={
                "type": "object",
                "properties": {
                    "hardware_status": {
                        "type": "object",
                        "description": "Hardware status information"
                    },
                    "error_codes": {
                        "type": "string",
                        "description": "Optional error codes from the hardware"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Optional temperature reading"
                    },
                    "performance_metrics": {
                        "type": "string",
                        "description": "Optional performance metrics JSON"
                    }
                },
                "required": ["hardware_status"]
            }
        ),
        mcp_types.Tool(
            name="predict_hardware_failure",
            description="Predict hardware failure probability using historical data and trends",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "The ID of the tower to analyze"
                    },
                    "component_type": {
                        "type": "string",
                        "description": "Optional specific component to analyze"
                    },
                    "prediction_horizon_hours": {
                        "type": "integer",
                        "description": "Hours ahead to predict (default: 168 = 1 week)",
                        "default": 168
                    }
                },
                "required": ["tower_id"]
            }
        ),
        mcp_types.Tool(
            name="schedule_proactive_maintenance",
            description="Schedule proactive maintenance based on predictive analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "The ID of the tower"
                    },
                    "component_type": {
                        "type": "string",
                        "description": "Type of component requiring maintenance"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority level (LOW, MEDIUM, HIGH, CRITICAL)"
                    },
                    "recommended_date": {
                        "type": "string",
                        "description": "Optional recommended maintenance date"
                    }
                },
                "required": ["tower_id", "component_type", "priority"]
            }
        ),
        mcp_types.Tool(
            name="get_equipment_lifecycle_status",
            description="Get equipment lifecycle status and recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "Optional filter by tower ID"
                    },
                    "component_type": {
                        "type": "string",
                        "description": "Optional filter by component type"
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="notify_field_team",
            description="Enhanced field team notification system with multiple channels and escalation",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "ID of the maintenance ticket"
                    },
                    "notification_type": {
                        "type": "string",
                        "description": "Type of notification (ASSIGNMENT, ESCALATION, COMPLETION, UPDATE)",
                        "default": "ASSIGNMENT"
                    },
                    "urgency_level": {
                        "type": "string",
                        "description": "Urgency level (LOW, NORMAL, HIGH, CRITICAL)",
                        "default": "NORMAL"
                    }
                },
                "required": ["ticket_id"]
            }
        ),
        mcp_types.Tool(
            name="auto_assign_ticket",
            description="Automatically assign a maintenance ticket to an available technician",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "ID of the ticket to assign"
                    },
                    "assignment_criteria": {
                        "type": "string",
                        "description": "Criteria for assignment (workload, location, expertise)",
                        "default": "workload"
                    }
                },
                "required": ["ticket_id"]
            }
        ),
        mcp_types.Tool(
            name="escalate_ticket",
            description="Escalate a maintenance ticket to higher priority or different team",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "ID of the ticket to escalate"
                    },
                    "escalation_reason": {
                        "type": "string",
                        "description": "Reason for escalation"
                    },
                    "escalate_to": {
                        "type": "string",
                        "description": "Who to escalate to (supervisor, specialist, emergency)",
                        "default": "supervisor"
                    }
                },
                "required": ["ticket_id", "escalation_reason"]
            }
        ),
        mcp_types.Tool(
            name="get_ticket_metrics",
            description="Get maintenance ticket metrics and analytics",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_period_hours": {
                        "type": "integer",
                        "description": "Time period for metrics calculation",
                        "default": 24
                    }
                }
            }
        )
    ]
    
    return tools

@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """MCP handler to execute a tool call requested by an MCP client."""
    logging.info(f"Maintenance MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    try:
        if name == "monitor_hardware_health":
            result = monitor_hardware_health(arguments.get("tower_id"))
        elif name == "detect_hardware_anomalies":
            threshold_hours = arguments.get("threshold_hours", 24)
            result = detect_hardware_anomalies(threshold_hours)
        elif name == "update_hardware_status":
            result = update_hardware_status(
                arguments.get("tower_id"),
                arguments.get("component_type"),
                arguments.get("status"),
                arguments.get("error_codes", ""),
                arguments.get("performance_metrics", ""),
                arguments.get("temperature")
            )
        elif name == "get_equipment_status_summary":
            result = get_equipment_status_summary(arguments.get("dummy_param", ""))
        elif name == "create_maintenance_ticket":
            result = create_maintenance_ticket(
                arguments.get("tower_id"),
                arguments.get("issue_description"),
                arguments.get("priority", "MEDIUM"),
                arguments.get("assigned_to")
            )
        elif name == "get_maintenance_tickets":
            result = get_maintenance_tickets(
                arguments.get("tower_id"),
                arguments.get("status"),
                arguments.get("priority"),
                arguments.get("limit", 50)
            )
        elif name == "update_maintenance_ticket":
            result = update_maintenance_ticket(
                arguments.get("ticket_id"),
                arguments.get("status"),
                arguments.get("assigned_to"),
                arguments.get("resolution_notes")
            )
        elif name == "classify_issue_priority":
            result = classify_issue_priority(
                arguments.get("hardware_status"),
                arguments.get("error_codes"),
                arguments.get("temperature"),
                arguments.get("performance_metrics")
            )
        elif name == "predict_hardware_failure":
            result = predict_hardware_failure(
                arguments.get("tower_id"),
                arguments.get("component_type"),
                arguments.get("prediction_horizon_hours", 168)
            )
        elif name == "schedule_proactive_maintenance":
            result = schedule_proactive_maintenance(
                arguments.get("tower_id"),
                arguments.get("component_type"),
                arguments.get("priority"),
                arguments.get("recommended_date")
            )
        elif name == "get_equipment_lifecycle_status":
            result = get_equipment_lifecycle_status(
                arguments.get("tower_id"),
                arguments.get("component_type")
            )
        elif name == "notify_field_team":
            result = notify_field_team(
                arguments.get("ticket_id"),
                arguments.get("notification_type", "ASSIGNMENT"),
                arguments.get("urgency_level", "NORMAL")
            )
        elif name == "auto_assign_ticket":
            result = auto_assign_ticket(
                arguments.get("ticket_id"),
                arguments.get("assignment_criteria", "workload")
            )
        elif name == "escalate_ticket":
            result = escalate_ticket(
                arguments.get("ticket_id"),
                arguments.get("escalation_reason"),
                arguments.get("escalate_to", "supervisor")
            )
        elif name == "get_ticket_metrics":
            result = get_ticket_metrics(arguments.get("time_period_hours", 24))
        else:
            result = {
                "success": False,
                "message": f"Tool '{name}' not implemented by this server.",
                "available_tools": [
                    "monitor_hardware_health", "detect_hardware_anomalies", "update_hardware_status",
                    "get_equipment_status_summary", "create_maintenance_ticket", "get_maintenance_tickets",
                    "update_maintenance_ticket", "classify_issue_priority", "predict_hardware_failure",
                    "schedule_proactive_maintenance", "get_equipment_lifecycle_status", "notify_field_team",
                    "auto_assign_ticket", "escalate_ticket", "get_ticket_metrics"
                ]
            }
        
        logging.info(f"Maintenance MCP Server: Tool '{name}' executed successfully")
        response_text = json.dumps(result, indent=2)
        return [mcp_types.TextContent(type="text", text=response_text)]

    except Exception as e:
        logging.error(f"Maintenance MCP Server: Error executing tool '{name}': {e}", exc_info=True)
        error_payload = {
            "success": False,
            "message": f"Failed to execute tool '{name}': {str(e)}",
            "tool_name": name
        }
        error_text = json.dumps(error_payload)
        return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    """Runs the MCP server, listening for connections over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logging.info("Maintenance MCP Stdio Server: Starting handshake with client...")
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
        logging.info("Maintenance MCP Stdio Server: Run loop finished or client disconnected.")

if __name__ == "__main__":
    logging.info("Launching Maintenance MCP Server via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("\nMaintenance MCP Server (stdio) stopped by user.")
    except Exception as e:
        logging.critical(
            f"Maintenance MCP Server (stdio) encountered an unhandled error: {e}", exc_info=True
        )
    finally:
        logging.info("Maintenance MCP Server (stdio) process exiting.")