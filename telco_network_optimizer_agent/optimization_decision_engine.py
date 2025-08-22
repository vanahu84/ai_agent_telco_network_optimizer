"""
Network Optimization Decision Engine
Implements central decision-making logic for network optimization
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationAction(Enum):
    """Types of optimization actions"""
    SPECTRUM_REALLOCATION = "SPECTRUM_REALLOCATION"
    LOAD_BALANCING = "LOAD_BALANCING"
    TRAFFIC_REDIRECT = "TRAFFIC_REDIRECT"
    MAINTENANCE_TRIGGER = "MAINTENANCE_TRIGGER"
    NO_ACTION = "NO_ACTION"

class SeverityLevel(Enum):
    """Severity levels for events and decisions"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class OptimizationStatus(Enum):
    """Status of optimization actions"""
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class TowerMetrics:
    """Tower performance metrics"""
    tower_id: str
    timestamp: datetime
    cpu_utilization: float
    memory_usage: float
    bandwidth_usage: float
    active_connections: int
    signal_strength: float
    error_rate: float

@dataclass
class CongestionEvent:
    """Congestion event data"""
    tower_id: str
    severity: SeverityLevel
    detected_at: datetime
    metrics: TowerMetrics
    affected_area: str

@dataclass
class OptimizationDecision:
    """Optimization decision result"""
    action: OptimizationAction
    priority: SeverityLevel
    affected_towers: List[str]
    parameters: Dict[str, Any]
    expected_improvement: float
    execution_time_estimate: int
    fallback_actions: List[OptimizationAction]

@dataclass
class OptimizationResult:
    """Result of optimization execution"""
    decision_id: str
    action: OptimizationAction
    status: OptimizationStatus
    effectiveness_score: float
    execution_time: int
    error_message: Optional[str] = None

class OptimizationDecisionEngine:
    """
    Central decision-making engine for network optimization
    Implements multi-criteria optimization algorithms and escalation procedures
    """
    
    def __init__(self, db_path: str = "telecom.db"):
        self.db_path = db_path
        self.optimization_thresholds = {
            'congestion_low': 80.0,
            'congestion_medium': 85.0,
            'congestion_high': 95.0,
            'prediction_confidence_min': 0.7,
            'maintenance_critical_threshold': 0.9,
            'sla_packet_loss_max': 1.0,
            'optimization_timeout': 120  # seconds
        }
        
        # Optimization weights for multi-criteria decision making
        self.optimization_weights = {
            'performance_impact': 0.4,
            'resource_efficiency': 0.3,
            'sla_compliance': 0.2,
            'execution_complexity': 0.1
        }
        
    async def analyze_network_state(self) -> Dict[str, Any]:
        """
        Analyze current network state across all towers
        Returns comprehensive network status for decision making
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current tower metrics (last 10 minutes)
            cutoff_time = (datetime.now() - timedelta(minutes=10)).isoformat()
            cursor.execute("""
                SELECT tower_id, timestamp, cpu_utilization, memory_usage, 
                       bandwidth_usage, active_connections, signal_strength, error_rate
                FROM tower_metrics 
                WHERE timestamp > ?
                ORDER BY tower_id, timestamp DESC
            """, (cutoff_time,))
            
            metrics_data = cursor.fetchall()
            
            # Get active congestion events
            cursor.execute("""
                SELECT tower_id, severity, detected_at, metrics_snapshot, affected_area
                FROM congestion_events 
                WHERE resolved_at IS NULL
                ORDER BY detected_at DESC
            """, )
            
            congestion_data = cursor.fetchall()
            
            # Get traffic predictions
            cursor.execute("""
                SELECT tower_id, predicted_load, confidence_level, prediction_horizon_minutes
                FROM traffic_predictions 
                WHERE prediction_timestamp > datetime('now', '-1 hour')
                ORDER BY tower_id, prediction_timestamp DESC
            """)
            
            prediction_data = cursor.fetchall()
            
            # Get hardware status
            cursor.execute("""
                SELECT tower_id, component_type, status, last_checked
                FROM hardware_status 
                WHERE last_checked > datetime('now', '-1 hour')
                AND status IN ('WARNING', 'CRITICAL', 'FAILED')
            """)
            
            hardware_issues = cursor.fetchall()
            
            conn.close()
            
            # Process and structure the data
            network_state = {
                'tower_metrics': self._process_tower_metrics(metrics_data),
                'active_congestion': self._process_congestion_events(congestion_data),
                'traffic_predictions': self._process_predictions(prediction_data),
                'hardware_issues': self._process_hardware_issues(hardware_issues),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Network state analyzed: {len(network_state['tower_metrics'])} towers, "
                       f"{len(network_state['active_congestion'])} congestion events")
            
            return network_state
            
        except Exception as e:
            logger.error(f"Error analyzing network state: {e}")
            return {}
    
    async def make_optimization_decision(self, network_state: Dict[str, Any]) -> OptimizationDecision:
        """
        Central decision-making algorithm using multi-criteria optimization
        Analyzes network state and determines optimal action
        """
        try:
            # Extract key metrics for decision making
            tower_metrics = network_state.get('tower_metrics', {})
            active_congestion = network_state.get('active_congestion', [])
            predictions = network_state.get('traffic_predictions', {})
            hardware_issues = network_state.get('hardware_issues', [])
            
            # Multi-criteria analysis
            decision_factors = await self._analyze_decision_factors(
                tower_metrics, active_congestion, predictions, hardware_issues
            )
            
            # Determine primary action based on weighted criteria
            primary_action = await self._select_primary_action(decision_factors)
            
            # Calculate priority and affected towers
            priority = self._calculate_priority(decision_factors)
            affected_towers = self._identify_affected_towers(decision_factors, primary_action)
            
            # Generate action parameters
            parameters = await self._generate_action_parameters(
                primary_action, affected_towers, decision_factors
            )
            
            # Estimate expected improvement and execution time
            expected_improvement = self._estimate_improvement(primary_action, decision_factors)
            execution_time = self._estimate_execution_time(primary_action, len(affected_towers))
            
            # Define fallback actions
            fallback_actions = self._define_fallback_actions(primary_action, decision_factors)
            
            decision = OptimizationDecision(
                action=primary_action,
                priority=priority,
                affected_towers=affected_towers,
                parameters=parameters,
                expected_improvement=expected_improvement,
                execution_time_estimate=execution_time,
                fallback_actions=fallback_actions
            )
            
            logger.info(f"Optimization decision made: {primary_action.value} for towers {affected_towers}")
            return decision
            
        except Exception as e:
            logger.error(f"Error making optimization decision: {e}")
            # Return safe fallback decision
            return OptimizationDecision(
                action=OptimizationAction.NO_ACTION,
                priority=SeverityLevel.LOW,
                affected_towers=[],
                parameters={},
                expected_improvement=0.0,
                execution_time_estimate=0,
                fallback_actions=[]
            )
    
    async def execute_optimization_with_fallback(self, decision: OptimizationDecision) -> OptimizationResult:
        """
        Execute optimization decision with escalation and fallback procedures
        """
        decision_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            # Log optimization attempt
            await self._log_optimization_attempt(decision_id, decision)
            
            # Execute primary action
            result = await self._execute_primary_action(decision_id, decision)
            
            # If primary action fails, try fallback actions
            if result.status == OptimizationStatus.FAILED and decision.fallback_actions:
                logger.warning(f"Primary action {decision.action.value} failed, trying fallbacks")
                result = await self._execute_fallback_actions(decision_id, decision)
            
            # Calculate final effectiveness score
            if result.status == OptimizationStatus.COMPLETED:
                result.effectiveness_score = await self._calculate_effectiveness_score(
                    decision_id, decision, result
                )
            
            execution_time = int((datetime.now() - start_time).total_seconds())
            result.execution_time = execution_time
            
            # Log final result
            await self._log_optimization_result(decision_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing optimization {decision_id}: {e}")
            return OptimizationResult(
                decision_id=decision_id,
                action=decision.action,
                status=OptimizationStatus.FAILED,
                effectiveness_score=0.0,
                execution_time=int((datetime.now() - start_time).total_seconds()),
                error_message=str(e)
            )
    
    async def check_optimization_triggers(self) -> bool:
        """
        Check if automated optimization should be triggered
        Returns True if optimization is needed
        """
        try:
            network_state = await self.analyze_network_state()
            
            # Check for immediate triggers
            triggers = {
                'high_congestion': self._check_congestion_trigger(network_state),
                'predicted_overload': self._check_prediction_trigger(network_state),
                'hardware_critical': self._check_hardware_trigger(network_state),
                'sla_violation': self._check_sla_trigger(network_state)
            }
            
            # Log trigger status
            active_triggers = [k for k, v in triggers.items() if v]
            if active_triggers:
                logger.info(f"Optimization triggers active: {active_triggers}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking optimization triggers: {e}")
            return False
    
    # Helper methods for decision making
    
    def _process_tower_metrics(self, metrics_data: List[Tuple]) -> Dict[str, TowerMetrics]:
        """Process raw tower metrics data into structured format"""
        tower_metrics = {}
        for row in metrics_data:
            tower_id = row[0]
            if tower_id not in tower_metrics:  # Take most recent metrics per tower
                tower_metrics[tower_id] = TowerMetrics(
                    tower_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    cpu_utilization=row[2],
                    memory_usage=row[3],
                    bandwidth_usage=row[4],
                    active_connections=row[5],
                    signal_strength=row[6],
                    error_rate=row[7]
                )
        return tower_metrics
    
    def _process_congestion_events(self, congestion_data: List[Tuple]) -> List[CongestionEvent]:
        """Process raw congestion data into structured format"""
        events = []
        for row in congestion_data:
            # Parse metrics snapshot if available
            metrics_snapshot = {}
            if row[3]:
                try:
                    metrics_snapshot = json.loads(row[3])
                except:
                    pass
            
            # Create dummy metrics from snapshot or defaults
            metrics = TowerMetrics(
                tower_id=row[0],
                timestamp=datetime.fromisoformat(row[2]),
                cpu_utilization=metrics_snapshot.get('cpu', 0),
                memory_usage=metrics_snapshot.get('memory', 0),
                bandwidth_usage=metrics_snapshot.get('bandwidth', 0),
                active_connections=metrics_snapshot.get('connections', 0),
                signal_strength=metrics_snapshot.get('signal', -70),
                error_rate=metrics_snapshot.get('error_rate', 0)
            )
            
            events.append(CongestionEvent(
                tower_id=row[0],
                severity=SeverityLevel(row[1]),
                detected_at=datetime.fromisoformat(row[2]),
                metrics=metrics,
                affected_area=row[4] or "Unknown"
            ))
        return events
    
    def _process_predictions(self, prediction_data: List[Tuple]) -> Dict[str, Dict]:
        """Process traffic prediction data"""
        predictions = {}
        for row in prediction_data:
            tower_id = row[0]
            if tower_id not in predictions:
                predictions[tower_id] = {
                    'predicted_load': row[1],
                    'confidence': row[2],
                    'horizon_minutes': row[3]
                }
        return predictions
    
    def _process_hardware_issues(self, hardware_data: List[Tuple]) -> List[Dict]:
        """Process hardware issue data"""
        issues = []
        for row in hardware_data:
            issues.append({
                'tower_id': row[0],
                'component_type': row[1],
                'status': row[2],
                'last_checked': row[3]
            })
        return issues
    
    async def _analyze_decision_factors(self, tower_metrics: Dict, congestion_events: List,
                                      predictions: Dict, hardware_issues: List) -> Dict[str, Any]:
        """Analyze all factors that influence optimization decisions"""
        factors = {
            'congestion_severity': 0.0,
            'prediction_risk': 0.0,
            'hardware_risk': 0.0,
            'sla_risk': 0.0,
            'affected_tower_count': 0,
            'peak_utilization': 0.0,
            'critical_towers': []
        }
        
        # Analyze congestion severity
        if congestion_events:
            severity_scores = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
            max_severity = max(severity_scores.get(event.severity.value, 0) for event in congestion_events)
            factors['congestion_severity'] = max_severity / 4.0
            factors['affected_tower_count'] = len(set(event.tower_id for event in congestion_events))
        
        # Analyze prediction risk
        high_risk_predictions = 0
        for tower_id, pred in predictions.items():
            if pred['predicted_load'] > 85 and pred['confidence'] > 0.7:
                high_risk_predictions += 1
        factors['prediction_risk'] = min(high_risk_predictions / max(len(predictions), 1), 1.0)
        
        # Analyze hardware risk
        critical_hardware = [issue for issue in hardware_issues if issue['status'] in ['CRITICAL', 'FAILED']]
        factors['hardware_risk'] = min(len(critical_hardware) / max(len(hardware_issues), 1), 1.0)
        
        # Analyze current utilization
        if tower_metrics:
            utilizations = [metrics.bandwidth_usage for metrics in tower_metrics.values()]
            factors['peak_utilization'] = max(utilizations) if utilizations else 0.0
            factors['critical_towers'] = [
                tower_id for tower_id, metrics in tower_metrics.items()
                if metrics.bandwidth_usage > 90 or metrics.cpu_utilization > 90
            ]
        
        # Calculate SLA risk based on multiple factors
        factors['sla_risk'] = min(
            (factors['congestion_severity'] * 0.4 + 
             factors['prediction_risk'] * 0.3 + 
             factors['hardware_risk'] * 0.3), 1.0
        )
        
        return factors
    
    async def _select_primary_action(self, decision_factors: Dict[str, Any]) -> OptimizationAction:
        """Select primary optimization action based on weighted decision factors"""
        
        # Decision matrix based on factors
        if decision_factors['hardware_risk'] > 0.7:
            return OptimizationAction.MAINTENANCE_TRIGGER
        
        if decision_factors['congestion_severity'] > 0.75:  # HIGH or CRITICAL
            if decision_factors['affected_tower_count'] > 1:
                return OptimizationAction.LOAD_BALANCING
            else:
                return OptimizationAction.SPECTRUM_REALLOCATION
        
        if decision_factors['prediction_risk'] > 0.6:
            return OptimizationAction.SPECTRUM_REALLOCATION
        
        if decision_factors['peak_utilization'] > 85:
            return OptimizationAction.TRAFFIC_REDIRECT
        
        return OptimizationAction.NO_ACTION
    
    def _calculate_priority(self, decision_factors: Dict[str, Any]) -> SeverityLevel:
        """Calculate optimization priority based on decision factors"""
        risk_score = (
            decision_factors['congestion_severity'] * 0.4 +
            decision_factors['sla_risk'] * 0.3 +
            decision_factors['hardware_risk'] * 0.2 +
            decision_factors['prediction_risk'] * 0.1
        )
        
        if risk_score > 0.75:
            return SeverityLevel.CRITICAL
        elif risk_score > 0.5:
            return SeverityLevel.HIGH
        elif risk_score > 0.25:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _identify_affected_towers(self, decision_factors: Dict[str, Any], 
                                action: OptimizationAction) -> List[str]:
        """Identify towers affected by the optimization action"""
        affected_towers = decision_factors.get('critical_towers', [])
        
        if action == OptimizationAction.LOAD_BALANCING:
            # For load balancing, include neighboring towers
            # This is a simplified version - in reality would use geographic data
            if len(affected_towers) < 2:
                affected_towers.extend(['TOWER_001', 'TOWER_002'])  # Default neighbors
        
        return list(set(affected_towers))  # Remove duplicates
    
    async def _generate_action_parameters(self, action: OptimizationAction, 
                                        affected_towers: List[str], 
                                        decision_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific parameters for the optimization action"""
        parameters = {
            'action_type': action.value,
            'towers': affected_towers,
            'timestamp': datetime.now().isoformat()
        }
        
        if action == OptimizationAction.SPECTRUM_REALLOCATION:
            parameters.update({
                'target_utilization': 75.0,
                'reallocation_percentage': min(20.0, decision_factors['peak_utilization'] - 75.0),
                'frequency_bands': ['2.6GHz', '3.5GHz']
            })
        
        elif action == OptimizationAction.LOAD_BALANCING:
            parameters.update({
                'balancing_algorithm': 'weighted_round_robin',
                'target_distribution': 'even',
                'max_redirect_percentage': 30.0
            })
        
        elif action == OptimizationAction.TRAFFIC_REDIRECT:
            parameters.update({
                'redirect_percentage': min(25.0, decision_factors['peak_utilization'] - 80.0),
                'target_towers': affected_towers[:2] if len(affected_towers) > 1 else ['TOWER_001']
            })
        
        elif action == OptimizationAction.MAINTENANCE_TRIGGER:
            parameters.update({
                'priority': 'HIGH' if decision_factors['hardware_risk'] > 0.8 else 'MEDIUM',
                'estimated_downtime': 120  # minutes
            })
        
        return parameters
    
    def _estimate_improvement(self, action: OptimizationAction, 
                            decision_factors: Dict[str, Any]) -> float:
        """Estimate expected improvement percentage from optimization action"""
        base_improvements = {
            OptimizationAction.SPECTRUM_REALLOCATION: 15.0,
            OptimizationAction.LOAD_BALANCING: 25.0,
            OptimizationAction.TRAFFIC_REDIRECT: 20.0,
            OptimizationAction.MAINTENANCE_TRIGGER: 10.0,
            OptimizationAction.NO_ACTION: 0.0
        }
        
        base_improvement = base_improvements.get(action, 0.0)
        
        # Adjust based on severity
        severity_multiplier = 1.0 + (decision_factors['congestion_severity'] * 0.5)
        
        return min(base_improvement * severity_multiplier, 50.0)  # Cap at 50%
    
    def _estimate_execution_time(self, action: OptimizationAction, tower_count: int) -> int:
        """Estimate execution time in seconds"""
        base_times = {
            OptimizationAction.SPECTRUM_REALLOCATION: 60,
            OptimizationAction.LOAD_BALANCING: 90,
            OptimizationAction.TRAFFIC_REDIRECT: 45,
            OptimizationAction.MAINTENANCE_TRIGGER: 30,
            OptimizationAction.NO_ACTION: 0
        }
        
        base_time = base_times.get(action, 60)
        if action == OptimizationAction.NO_ACTION:
            return 0
        return base_time + ((tower_count - 1) * 15)  # Add 15 seconds per additional tower
    
    def _define_fallback_actions(self, primary_action: OptimizationAction, 
                               decision_factors: Dict[str, Any]) -> List[OptimizationAction]:
        """Define fallback actions if primary action fails"""
        fallback_map = {
            OptimizationAction.SPECTRUM_REALLOCATION: [OptimizationAction.LOAD_BALANCING, OptimizationAction.TRAFFIC_REDIRECT],
            OptimizationAction.LOAD_BALANCING: [OptimizationAction.TRAFFIC_REDIRECT, OptimizationAction.SPECTRUM_REALLOCATION],
            OptimizationAction.TRAFFIC_REDIRECT: [OptimizationAction.SPECTRUM_REALLOCATION],
            OptimizationAction.MAINTENANCE_TRIGGER: [OptimizationAction.TRAFFIC_REDIRECT],
            OptimizationAction.NO_ACTION: []
        }
        
        return fallback_map.get(primary_action, [])
    
    # Trigger checking methods
    
    def _check_congestion_trigger(self, network_state: Dict[str, Any]) -> bool:
        """Check if congestion levels trigger optimization"""
        congestion_events = network_state.get('active_congestion', [])
        return any(event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] 
                  for event in congestion_events)
    
    def _check_prediction_trigger(self, network_state: Dict[str, Any]) -> bool:
        """Check if traffic predictions trigger optimization"""
        predictions = network_state.get('traffic_predictions', {})
        return any(pred['predicted_load'] > 85 and pred['confidence'] > 0.7 
                  for pred in predictions.values())
    
    def _check_hardware_trigger(self, network_state: Dict[str, Any]) -> bool:
        """Check if hardware issues trigger optimization"""
        hardware_issues = network_state.get('hardware_issues', [])
        return any(issue['status'] in ['CRITICAL', 'FAILED'] for issue in hardware_issues)
    
    def _check_sla_trigger(self, network_state: Dict[str, Any]) -> bool:
        """Check if SLA violations trigger optimization"""
        tower_metrics = network_state.get('tower_metrics', {})
        return any(metrics.error_rate > 1.0 or metrics.bandwidth_usage > 95 
                  for metrics in tower_metrics.values())
    
    # Execution methods (simplified implementations)
    
    async def _execute_primary_action(self, decision_id: str, 
                                    decision: OptimizationDecision) -> OptimizationResult:
        """Execute the primary optimization action"""
        try:
            # Simulate action execution based on type
            await asyncio.sleep(1)  # Simulate processing time
            
            # For now, simulate successful execution
            # In real implementation, this would call actual MCP servers
            
            return OptimizationResult(
                decision_id=decision_id,
                action=decision.action,
                status=OptimizationStatus.COMPLETED,
                effectiveness_score=0.0,  # Will be calculated later
                execution_time=0
            )
            
        except Exception as e:
            return OptimizationResult(
                decision_id=decision_id,
                action=decision.action,
                status=OptimizationStatus.FAILED,
                effectiveness_score=0.0,
                execution_time=0,
                error_message=str(e)
            )
    
    async def _execute_fallback_actions(self, decision_id: str, 
                                      decision: OptimizationDecision) -> OptimizationResult:
        """Execute fallback actions if primary action fails"""
        for fallback_action in decision.fallback_actions:
            try:
                logger.info(f"Trying fallback action: {fallback_action.value}")
                await asyncio.sleep(0.5)  # Simulate processing
                
                # Simulate successful fallback
                return OptimizationResult(
                    decision_id=decision_id,
                    action=fallback_action,
                    status=OptimizationStatus.COMPLETED,
                    effectiveness_score=0.0,
                    execution_time=0
                )
                
            except Exception as e:
                logger.warning(f"Fallback action {fallback_action.value} failed: {e}")
                continue
        
        # All fallbacks failed
        return OptimizationResult(
            decision_id=decision_id,
            action=decision.action,
            status=OptimizationStatus.FAILED,
            effectiveness_score=0.0,
            execution_time=0,
            error_message="All fallback actions failed"
        )
    
    async def _calculate_effectiveness_score(self, decision_id: str, 
                                           decision: OptimizationDecision,
                                           result: OptimizationResult) -> float:
        """Calculate effectiveness score of the optimization"""
        # Simplified effectiveness calculation
        # In real implementation, would compare before/after metrics
        return min(decision.expected_improvement * 0.8, 100.0)
    
    async def _log_optimization_attempt(self, decision_id: str, decision: OptimizationDecision):
        """Log optimization attempt to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO optimization_actions 
                (action_type, tower_ids, parameters, executed_at, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                decision.action.value,
                json.dumps(decision.affected_towers),
                json.dumps(decision.parameters),
                datetime.now().isoformat(),
                OptimizationStatus.PENDING.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging optimization attempt: {e}")
    
    async def _log_optimization_result(self, decision_id: str, result: OptimizationResult):
        """Log optimization result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE optimization_actions 
                SET status = ?, effectiveness_score = ?
                WHERE id = (
                    SELECT id FROM optimization_actions 
                    WHERE action_type = ? AND executed_at > datetime('now', '-1 hour')
                    ORDER BY executed_at DESC LIMIT 1
                )
            """, (
                result.status.value,
                result.effectiveness_score,
                result.action.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging optimization result: {e}")