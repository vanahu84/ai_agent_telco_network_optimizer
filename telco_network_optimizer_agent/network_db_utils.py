"""
Database Utility Functions for Network Optimization System
Provides high-level database operations for network optimization data
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import json

from network_models import (
    Tower, TowerMetrics, CongestionEvent, SpectrumAllocation,
    OptimizationAction, TrafficForecast, HardwareComponent, MaintenanceTicket,
    SeverityLevel, TowerStatus, HardwareStatus, ComponentType, OptimizationActionType,
    create_tower_metrics_from_db_row, create_congestion_event_from_db_row,
    create_optimization_action_from_db_row
)

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")

class NetworkDatabaseManager:
    """Database manager for network optimization operations"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # Tower Management
    
    def get_all_towers(self) -> List[Tower]:
        """Get all towers from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM towers ORDER BY name")
            rows = cursor.fetchall()
            
            return [Tower(
                id=row['id'],
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                coverage_radius=row['coverage_radius'],
                max_capacity=row['max_capacity'],
                technology=row['technology'],
                status=TowerStatus(row['status'])
            ) for row in rows]
    
    def get_tower_by_id(self, tower_id: str) -> Optional[Tower]:
        """Get specific tower by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM towers WHERE id = ?", (tower_id,))
            row = cursor.fetchone()
            
            if row:
                return Tower(
                    id=row['id'],
                    name=row['name'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    coverage_radius=row['coverage_radius'],
                    max_capacity=row['max_capacity'],
                    technology=row['technology'],
                    status=TowerStatus(row['status'])
                )
            return None
    
    def get_active_towers(self) -> List[Tower]:
        """Get all active towers"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM towers WHERE status = 'ACTIVE' ORDER BY name")
            rows = cursor.fetchall()
            
            return [Tower(
                id=row['id'],
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                coverage_radius=row['coverage_radius'],
                max_capacity=row['max_capacity'],
                technology=row['technology'],
                status=TowerStatus(row['status'])
            ) for row in rows]
    
    # Tower Metrics Management
    
    def insert_tower_metrics(self, metrics: TowerMetrics) -> int:
        """Insert tower metrics into database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tower_metrics 
                (tower_id, timestamp, cpu_utilization, memory_usage, bandwidth_usage, 
                 active_connections, signal_strength, error_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.tower_id, metrics.timestamp.isoformat(),
                metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage,
                metrics.active_connections, metrics.signal_strength, metrics.error_rate
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_latest_tower_metrics(self, tower_id: str) -> Optional[TowerMetrics]:
        """Get the most recent metrics for a tower"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tower_metrics 
                WHERE tower_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (tower_id,))
            row = cursor.fetchone()
            
            if row:
                return TowerMetrics(
                    tower_id=row['tower_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    cpu_utilization=row['cpu_utilization'],
                    memory_usage=row['memory_usage'],
                    bandwidth_usage=row['bandwidth_usage'],
                    active_connections=row['active_connections'],
                    signal_strength=row['signal_strength'],
                    error_rate=row['error_rate']
                )
            return None
    
    def get_tower_metrics_history(self, tower_id: str, hours: int = 24) -> List[TowerMetrics]:
        """Get tower metrics history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tower_metrics 
                WHERE tower_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (tower_id, cutoff_time.isoformat()))
            rows = cursor.fetchall()
            
            return [TowerMetrics(
                tower_id=row['tower_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                cpu_utilization=row['cpu_utilization'],
                memory_usage=row['memory_usage'],
                bandwidth_usage=row['bandwidth_usage'],
                active_connections=row['active_connections'],
                signal_strength=row['signal_strength'],
                error_rate=row['error_rate']
            ) for row in rows]
    
    def get_congested_towers(self, threshold: float = 80.0) -> List[Tuple[str, TowerMetrics]]:
        """Get towers currently experiencing congestion"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT tower_id FROM tower_metrics tm1
                WHERE timestamp = (
                    SELECT MAX(timestamp) FROM tower_metrics tm2 
                    WHERE tm2.tower_id = tm1.tower_id
                ) AND (
                    cpu_utilization > ? OR 
                    memory_usage > ? OR 
                    bandwidth_usage > ?
                )
            """, (threshold, threshold, threshold))
            
            congested_towers = []
            for row in cursor.fetchall():
                tower_id = row['tower_id']
                metrics = self.get_latest_tower_metrics(tower_id)
                if metrics:
                    congested_towers.append((tower_id, metrics))
            
            return congested_towers
    
    # Congestion Event Management
    
    def insert_congestion_event(self, event: CongestionEvent) -> int:
        """Insert congestion event into database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO congestion_events 
                (tower_id, severity, detected_at, resolved_at, metrics_snapshot, affected_area)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.tower_id, event.severity.value, event.detected_at.isoformat(),
                event.resolved_at.isoformat() if event.resolved_at else None,
                json.dumps(event.metrics.to_dict()), event.affected_area
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_active_congestion_events(self) -> List[CongestionEvent]:
        """Get all unresolved congestion events"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM congestion_events 
                WHERE resolved_at IS NULL 
                ORDER BY detected_at DESC
            """)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                # Parse metrics snapshot - handle simplified format
                metrics_data = json.loads(row['metrics_snapshot'])
                
                # Create metrics object with available data, using defaults for missing fields
                metrics = TowerMetrics(
                    tower_id=row['tower_id'],  # Use tower_id from the event row
                    timestamp=datetime.fromisoformat(row['detected_at']),  # Use detection time
                    cpu_utilization=metrics_data.get('cpu', 0.0),
                    memory_usage=metrics_data.get('memory', 0.0),
                    bandwidth_usage=metrics_data.get('bandwidth', 0.0),
                    active_connections=metrics_data.get('active_connections', 0),
                    signal_strength=metrics_data.get('signal_strength', -70.0),
                    error_rate=metrics_data.get('error_rate', 0.0)
                )
                
                event = CongestionEvent(
                    id=row['id'],
                    tower_id=row['tower_id'],
                    severity=SeverityLevel(row['severity']),
                    detected_at=datetime.fromisoformat(row['detected_at']),
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                    metrics=metrics,
                    affected_area=row['affected_area']
                )
                events.append(event)
            
            return events
    
    def resolve_congestion_event(self, event_id: int) -> bool:
        """Mark congestion event as resolved"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE congestion_events 
                SET resolved_at = ? 
                WHERE id = ? AND resolved_at IS NULL
            """, (datetime.now().isoformat(), event_id))
            conn.commit()
            return cursor.rowcount > 0
    
    # Optimization Actions Management
    
    def insert_optimization_action(self, action: OptimizationAction) -> int:
        """Insert optimization action into database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO optimization_actions 
                (action_type, tower_ids, parameters, executed_at, effectiveness_score, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                action.action_type.value, action.tower_ids_str, action.parameters_json,
                action.executed_at.isoformat(), action.effectiveness_score, action.status.value
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_optimization_actions(self, hours: int = 24) -> List[OptimizationAction]:
        """Get recent optimization actions"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM optimization_actions 
                WHERE executed_at >= ?
                ORDER BY executed_at DESC
            """, (cutoff_time.isoformat(),))
            rows = cursor.fetchall()
            
            return [OptimizationAction(
                id=row['id'],
                action_type=OptimizationActionType(row['action_type']),
                tower_ids=row['tower_ids'].split(',') if row['tower_ids'] else [],
                parameters=json.loads(row['parameters']) if row['parameters'] else {},
                executed_at=datetime.fromisoformat(row['executed_at']),
                effectiveness_score=row['effectiveness_score'],
                status=row['status']
            ) for row in rows]
    
    # Traffic Predictions Management
    
    def insert_traffic_prediction(self, prediction: TrafficForecast) -> int:
        """Insert traffic prediction into database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO traffic_predictions 
                (tower_id, prediction_timestamp, predicted_load, confidence_level, prediction_horizon_minutes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction.tower_id, prediction.prediction_timestamp.isoformat(),
                prediction.predicted_load, prediction.confidence_level,
                prediction.horizon_minutes, prediction.created_at.isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_high_load_predictions(self, confidence_threshold: float = 0.7) -> List[TrafficForecast]:
        """Get predictions indicating high load with high confidence"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM traffic_predictions 
                WHERE predicted_load > 80 AND confidence_level > ?
                AND prediction_timestamp > ?
                ORDER BY prediction_timestamp ASC
            """, (confidence_threshold, datetime.now().isoformat()))
            rows = cursor.fetchall()
            
            return [TrafficForecast(
                id=row['id'],
                tower_id=row['tower_id'],
                prediction_timestamp=datetime.fromisoformat(row['prediction_timestamp']),
                predicted_load=row['predicted_load'],
                confidence_level=row['confidence_level'],
                horizon_minutes=row['prediction_horizon_minutes'],
                created_at=datetime.fromisoformat(row['created_at'])
            ) for row in rows]
    
    # Hardware Status Management
    
    def insert_hardware_status(self, hardware: HardwareComponent) -> int:
        """Insert hardware status into database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO hardware_status 
                (tower_id, component_type, status, last_checked, error_codes, performance_metrics, temperature)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                hardware.tower_id, hardware.component_type.value, hardware.status.value,
                hardware.last_checked.isoformat(), hardware.error_codes,
                hardware.performance_metrics_json, hardware.temperature
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_critical_hardware_issues(self) -> List[HardwareComponent]:
        """Get hardware components with critical issues"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM hardware_status 
                WHERE status IN ('CRITICAL', 'FAILED')
                ORDER BY last_checked DESC
            """)
            rows = cursor.fetchall()
            
            return [HardwareComponent(
                id=row['id'],
                tower_id=row['tower_id'],
                component_type=ComponentType(row['component_type']),
                status=HardwareStatus(row['status']),
                last_checked=datetime.fromisoformat(row['last_checked']),
                error_codes=row['error_codes'],
                performance_metrics=json.loads(row['performance_metrics']) if row['performance_metrics'] else None,
                temperature=row['temperature']
            ) for row in rows]
    
    # Maintenance Tickets Management
    
    def create_maintenance_ticket(self, ticket: MaintenanceTicket) -> int:
        """Create new maintenance ticket"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO maintenance_tickets 
                (tower_id, priority, issue_description, created_at, assigned_to, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticket.tower_id, ticket.priority.value, ticket.issue_description,
                ticket.created_at.isoformat(), ticket.assigned_to, ticket.status
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_open_maintenance_tickets(self) -> List[MaintenanceTicket]:
        """Get all open maintenance tickets"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM maintenance_tickets 
                WHERE status IN ('OPEN', 'ASSIGNED', 'IN_PROGRESS')
                ORDER BY 
                    CASE priority 
                        WHEN 'CRITICAL' THEN 1 
                        WHEN 'HIGH' THEN 2 
                        WHEN 'MEDIUM' THEN 3 
                        WHEN 'LOW' THEN 4 
                    END,
                    created_at ASC
            """)
            rows = cursor.fetchall()
            
            return [MaintenanceTicket(
                id=row['id'],
                tower_id=row['tower_id'],
                priority=SeverityLevel(row['priority']),
                issue_description=row['issue_description'],
                created_at=datetime.fromisoformat(row['created_at']),
                assigned_to=row['assigned_to'],
                status=row['status'],
                resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                resolution_notes=row['resolution_notes']
            ) for row in rows]
    
    # Network Events Logging
    
    def log_network_event(self, event_type: str, tower_id: Optional[str], 
                         severity: str, description: str, metadata: Optional[Dict] = None) -> int:
        """Log a network event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO network_events 
                (event_type, tower_id, timestamp, severity, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_type, tower_id, datetime.now().isoformat(),
                severity, description, json.dumps(metadata) if metadata else None
            ))
            conn.commit()
            return cursor.lastrowid
    
    # Analytics and Reporting
    
    def get_tower_performance_summary(self, tower_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a tower"""
        metrics_history = self.get_tower_metrics_history(tower_id, hours)
        
        if not metrics_history:
            return {"error": "No metrics found"}
        
        cpu_values = [m.cpu_utilization for m in metrics_history]
        memory_values = [m.memory_usage for m in metrics_history]
        bandwidth_values = [m.bandwidth_usage for m in metrics_history]
        
        return {
            "tower_id": tower_id,
            "period_hours": hours,
            "total_measurements": len(metrics_history),
            "cpu_utilization": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_usage": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "bandwidth_usage": {
                "avg": sum(bandwidth_values) / len(bandwidth_values),
                "max": max(bandwidth_values),
                "min": min(bandwidth_values)
            },
            "congestion_events": len([m for m in metrics_history if m.is_congested]),
            "latest_metrics": metrics_history[0].to_dict() if metrics_history else None
        }
    
    def get_network_health_overview(self) -> Dict[str, Any]:
        """Get overall network health overview"""
        towers = self.get_all_towers()
        active_towers = [t for t in towers if t.status == TowerStatus.ACTIVE]
        congested_towers = self.get_congested_towers()
        active_congestion_events = self.get_active_congestion_events()
        critical_hardware = self.get_critical_hardware_issues()
        open_tickets = self.get_open_maintenance_tickets()
        
        return {
            "total_towers": len(towers),
            "active_towers": len(active_towers),
            "congested_towers": len(congested_towers),
            "active_congestion_events": len(active_congestion_events),
            "critical_hardware_issues": len(critical_hardware),
            "open_maintenance_tickets": len(open_tickets),
            "network_health_score": self._calculate_health_score(
                len(active_towers), len(congested_towers), 
                len(active_congestion_events), len(critical_hardware)
            )
        }
    
    def _calculate_health_score(self, active_towers: int, congested_towers: int, 
                               congestion_events: int, critical_hardware: int) -> float:
        """Calculate overall network health score (0-100)"""
        if active_towers == 0:
            return 0.0
        
        congestion_penalty = (congested_towers / active_towers) * 30
        event_penalty = min(congestion_events * 5, 20)
        hardware_penalty = min(critical_hardware * 10, 25)
        
        health_score = 100 - congestion_penalty - event_penalty - hardware_penalty
        return max(0.0, min(100.0, health_score))

# Convenience functions for common operations

def get_db_manager() -> NetworkDatabaseManager:
    """Get database manager instance"""
    return NetworkDatabaseManager()

def initialize_sample_data():
    """Initialize database with sample data for testing"""
    from network_optimization_schema import migrate_database, populate_sample_data
    migrate_database()
    populate_sample_data()

if __name__ == "__main__":
    # Test database operations
    db_manager = get_db_manager()
    
    print("Testing database operations...")
    
    # Test tower retrieval
    towers = db_manager.get_all_towers()
    print(f"Found {len(towers)} towers")
    
    # Test metrics retrieval
    if towers:
        tower_id = towers[0].id
        metrics = db_manager.get_latest_tower_metrics(tower_id)
        if metrics:
            print(f"Latest metrics for {tower_id}: CPU={metrics.cpu_utilization}%, Bandwidth={metrics.bandwidth_usage}%")
        
        # Test performance summary
        summary = db_manager.get_tower_performance_summary(tower_id)
        print(f"Performance summary: {summary}")
    
    # Test network health overview
    health = db_manager.get_network_health_overview()
    print(f"Network health overview: {health}")
    
    print("Database operations test completed!")