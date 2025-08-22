"""
Core Data Models for Network Optimization System
Defines data structures for tower metrics, congestion events, and optimization plans
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import json

# Enums for type safety
class SeverityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class TowerStatus(Enum):
    ACTIVE = "ACTIVE"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"

class HardwareStatus(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"

class ComponentType(Enum):
    ANTENNA = "ANTENNA"
    PROCESSOR = "PROCESSOR"
    MEMORY = "MEMORY"
    POWER = "POWER"
    COOLING = "COOLING"
    NETWORK = "NETWORK"

class OptimizationActionType(Enum):
    SPECTRUM_REALLOCATION = "SPECTRUM_REALLOCATION"
    LOAD_BALANCING = "LOAD_BALANCING"
    TRAFFIC_REDIRECT = "TRAFFIC_REDIRECT"
    MAINTENANCE_TRIGGER = "MAINTENANCE_TRIGGER"

class ActionStatus(Enum):
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# Core Data Models

@dataclass
class Tower:
    """Represents a 5G tower in the network"""
    id: str
    name: str
    latitude: float
    longitude: float
    coverage_radius: float = 2.0
    max_capacity: int = 1000
    technology: str = "5G"
    status: TowerStatus = TowerStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'coverage_radius': self.coverage_radius,
            'max_capacity': self.max_capacity,
            'technology': self.technology,
            'status': self.status.value
        }

@dataclass
class TowerMetrics:
    """Real-time performance metrics for a tower"""
    tower_id: str
    timestamp: datetime
    cpu_utilization: float
    memory_usage: float
    bandwidth_usage: float
    active_connections: int
    signal_strength: float = -70.0
    error_rate: float = 0.0
    
    def __post_init__(self):
        # Validate metrics ranges
        if not (0 <= self.cpu_utilization <= 100):
            raise ValueError("CPU utilization must be between 0 and 100")
        if not (0 <= self.memory_usage <= 100):
            raise ValueError("Memory usage must be between 0 and 100")
        if not (0 <= self.bandwidth_usage <= 100):
            raise ValueError("Bandwidth usage must be between 0 and 100")
        if self.active_connections < 0:
            raise ValueError("Active connections cannot be negative")
        if not (0 <= self.error_rate <= 100):
            raise ValueError("Error rate must be between 0 and 100")
    
    @property
    def is_congested(self) -> bool:
        """Check if tower is experiencing congestion"""
        return (self.cpu_utilization > 80 or 
                self.memory_usage > 80 or 
                self.bandwidth_usage > 80)
    
    @property
    def congestion_severity(self) -> SeverityLevel:
        """Determine congestion severity level"""
        max_usage = max(self.cpu_utilization, self.memory_usage, self.bandwidth_usage)
        if max_usage >= 95:
            return SeverityLevel.HIGH
        elif max_usage >= 85:
            return SeverityLevel.MEDIUM
        elif max_usage >= 80:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tower_id': self.tower_id,
            'timestamp': self.timestamp.isoformat(),
            'cpu_utilization': self.cpu_utilization,
            'memory_usage': self.memory_usage,
            'bandwidth_usage': self.bandwidth_usage,
            'active_connections': self.active_connections,
            'signal_strength': self.signal_strength,
            'error_rate': self.error_rate,
            'is_congested': self.is_congested,
            'congestion_severity': self.congestion_severity.value
        }

@dataclass
class CongestionEvent:
    """Represents a network congestion event"""
    tower_id: str
    severity: SeverityLevel
    detected_at: datetime
    metrics: TowerMetrics
    affected_area: str
    resolved_at: Optional[datetime] = None
    id: Optional[int] = None
    
    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None
    
    @property
    def duration_minutes(self) -> Optional[int]:
        if self.resolved_at:
            return int((self.resolved_at - self.detected_at).total_seconds() / 60)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tower_id': self.tower_id,
            'severity': self.severity.value,
            'detected_at': self.detected_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'affected_area': self.affected_area,
            'metrics_snapshot': self.metrics.to_dict(),
            'is_resolved': self.is_resolved,
            'duration_minutes': self.duration_minutes
        }

@dataclass
class SpectrumAllocation:
    """Represents spectrum allocation for a tower"""
    tower_id: str
    frequency_band: str
    allocated_bandwidth: float
    allocation_timestamp: datetime
    utilization_percentage: float = 0.0
    expires_at: Optional[datetime] = None
    id: Optional[int] = None
    
    def __post_init__(self):
        if self.allocated_bandwidth <= 0:
            raise ValueError("Allocated bandwidth must be positive")
        if not (0 <= self.utilization_percentage <= 100):
            raise ValueError("Utilization percentage must be between 0 and 100")
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    @property
    def available_bandwidth(self) -> float:
        return self.allocated_bandwidth * (1 - self.utilization_percentage / 100)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tower_id': self.tower_id,
            'frequency_band': self.frequency_band,
            'allocated_bandwidth': self.allocated_bandwidth,
            'utilization_percentage': self.utilization_percentage,
            'allocation_timestamp': self.allocation_timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_expired': self.is_expired,
            'available_bandwidth': self.available_bandwidth
        }

@dataclass
class OptimizationAction:
    """Represents a network optimization action"""
    action_type: OptimizationActionType
    tower_ids: List[str]
    parameters: Dict[str, Any]
    executed_at: datetime
    effectiveness_score: Optional[float] = None
    status: ActionStatus = ActionStatus.PENDING
    id: Optional[int] = None
    
    def __post_init__(self):
        if self.effectiveness_score is not None:
            if not (0 <= self.effectiveness_score <= 100):
                raise ValueError("Effectiveness score must be between 0 and 100")
    
    @property
    def tower_ids_str(self) -> str:
        """Convert tower IDs list to comma-separated string for database storage"""
        return ','.join(self.tower_ids)
    
    @property
    def parameters_json(self) -> str:
        """Convert parameters dict to JSON string for database storage"""
        return json.dumps(self.parameters)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'action_type': self.action_type.value,
            'tower_ids': self.tower_ids,
            'parameters': self.parameters,
            'executed_at': self.executed_at.isoformat(),
            'effectiveness_score': self.effectiveness_score,
            'status': self.status.value
        }

@dataclass
class OptimizationPlan:
    """Comprehensive optimization plan for network improvements"""
    plan_id: str
    affected_towers: List[str]
    actions: List[OptimizationAction]
    expected_improvement: float
    execution_time_estimate: int  # in minutes
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def total_actions(self) -> int:
        return len(self.actions)
    
    @property
    def estimated_completion(self) -> datetime:
        from datetime import timedelta
        return self.created_at + timedelta(minutes=self.execution_time_estimate)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plan_id': self.plan_id,
            'affected_towers': self.affected_towers,
            'actions': [action.to_dict() for action in self.actions],
            'expected_improvement': self.expected_improvement,
            'execution_time_estimate': self.execution_time_estimate,
            'created_at': self.created_at.isoformat(),
            'total_actions': self.total_actions,
            'estimated_completion': self.estimated_completion.isoformat()
        }

@dataclass
class TrafficForecast:
    """Traffic prediction for a tower"""
    tower_id: str
    prediction_timestamp: datetime
    predicted_load: float
    confidence_level: float
    horizon_minutes: int = 60
    created_at: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None
    
    def __post_init__(self):
        if not (0 <= self.predicted_load <= 100):
            raise ValueError("Predicted load must be between 0 and 100")
        if not (0 <= self.confidence_level <= 1.0):
            raise ValueError("Confidence level must be between 0 and 1")
    
    @property
    def is_high_load_predicted(self) -> bool:
        return self.predicted_load > 80 and self.confidence_level > 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tower_id': self.tower_id,
            'prediction_timestamp': self.prediction_timestamp.isoformat(),
            'predicted_load': self.predicted_load,
            'confidence_level': self.confidence_level,
            'horizon_minutes': self.horizon_minutes,
            'created_at': self.created_at.isoformat(),
            'is_high_load_predicted': self.is_high_load_predicted
        }

@dataclass
class HardwareComponent:
    """Hardware component status for a tower"""
    tower_id: str
    component_type: ComponentType
    status: HardwareStatus
    last_checked: datetime
    error_codes: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    id: Optional[int] = None
    
    @property
    def needs_attention(self) -> bool:
        return self.status in [HardwareStatus.WARNING, HardwareStatus.CRITICAL, HardwareStatus.FAILED]
    
    @property
    def performance_metrics_json(self) -> str:
        """Convert performance metrics to JSON string for database storage"""
        return json.dumps(self.performance_metrics) if self.performance_metrics else ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tower_id': self.tower_id,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'last_checked': self.last_checked.isoformat(),
            'error_codes': self.error_codes,
            'performance_metrics': self.performance_metrics,
            'temperature': self.temperature,
            'needs_attention': self.needs_attention
        }

@dataclass
class MaintenanceTicket:
    """Maintenance ticket for tower issues"""
    tower_id: str
    priority: SeverityLevel
    issue_description: str
    created_at: datetime
    assigned_to: Optional[str] = None
    status: str = "OPEN"
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    id: Optional[int] = None
    
    @property
    def is_resolved(self) -> bool:
        return self.status in ["RESOLVED", "CLOSED"]
    
    @property
    def resolution_time_hours(self) -> Optional[float]:
        if self.resolved_at:
            return (self.resolved_at - self.created_at).total_seconds() / 3600
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tower_id': self.tower_id,
            'priority': self.priority.value,
            'issue_description': self.issue_description,
            'created_at': self.created_at.isoformat(),
            'assigned_to': self.assigned_to,
            'status': self.status,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes,
            'is_resolved': self.is_resolved,
            'resolution_time_hours': self.resolution_time_hours
        }

# Utility functions for model creation

def create_tower_metrics_from_db_row(row: tuple) -> TowerMetrics:
    """Create TowerMetrics object from database row"""
    return TowerMetrics(
        tower_id=row[1],
        timestamp=datetime.fromisoformat(row[2]),
        cpu_utilization=row[3],
        memory_usage=row[4],
        bandwidth_usage=row[5],
        active_connections=row[6],
        signal_strength=row[7],
        error_rate=row[8]
    )

def create_congestion_event_from_db_row(row: tuple, metrics: TowerMetrics) -> CongestionEvent:
    """Create CongestionEvent object from database row"""
    return CongestionEvent(
        id=row[0],
        tower_id=row[1],
        severity=SeverityLevel(row[2]),
        detected_at=datetime.fromisoformat(row[3]),
        resolved_at=datetime.fromisoformat(row[4]) if row[4] else None,
        metrics=metrics,
        affected_area=row[6] if len(row) > 6 else ""
    )

def create_optimization_action_from_db_row(row: tuple) -> OptimizationAction:
    """Create OptimizationAction object from database row"""
    return OptimizationAction(
        id=row[0],
        action_type=OptimizationActionType(row[1]),
        tower_ids=row[2].split(',') if row[2] else [],
        parameters=json.loads(row[3]) if row[3] else {},
        executed_at=datetime.fromisoformat(row[4]),
        effectiveness_score=row[5],
        status=ActionStatus(row[6]) if len(row) > 6 else ActionStatus.PENDING
    )