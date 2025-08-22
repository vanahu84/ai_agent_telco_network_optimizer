"""
Real-Time Tower Load MCP Server
Monitors and analyzes 5G tower performance metrics in real-time
Implements congestion detection and load analysis capabilities
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

# ADK Tool Imports
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import network models and database utilities
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
    
    def classify_congestion_severity(self, tower_id: str) -> Dict[str, Any]:
        """Classify current congestion severity for a tower"""
        try:
            metrics = self.db_manager.get_latest_tower_metrics(tower_id)
            if not metrics:
                return {
                    "success": False,
                    "message": f"No current metrics found for tower {tower_id}",
                    "tower_id": tower_id
                }
            
            severity = metrics.congestion_severity
            max_usage = max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage)
            
            # Determine recommended actions based on severity
            recommendations = []
            if severity == SeverityLevel.HIGH:
                recommendations = [
                    "Immediate load balancing required",
                    "Consider traffic redirection to neighboring towers",
                    "Alert network operations center",
                    "Prepare for emergency spectrum reallocation"
                ]
            elif severity == SeverityLevel.MEDIUM:
                recommendations = [
                    "Monitor closely for escalation",
                    "Prepare load balancing procedures",
                    "Check neighboring tower capacity",
                    "Consider proactive spectrum adjustment"
                ]
            elif severity == SeverityLevel.LOW:
                recommendations = [
                    "Continue monitoring",
                    "Log event for trend analysis",
                    "No immediate action required"
                ]
            
            result = {
                "success": True,
                "tower_id": tower_id,
                "current_severity": severity.value,
                "is_congested": metrics.is_congested,
                "max_resource_usage": max_usage,
                "resource_breakdown": {
                    "cpu_utilization": metrics.cpu_utilization,
                    "memory_usage": metrics.memory_usage,
                    "bandwidth_usage": metrics.bandwidth_usage
                },
                "active_connections": metrics.active_connections,
                "signal_strength": metrics.signal_strength,
                "error_rate": metrics.error_rate,
                "recommendations": recommendations,
                "classification_timestamp": datetime.now().isoformat(),
                "metrics_timestamp": metrics.timestamp.isoformat()
            }
            
            logging.info(f"Classified congestion severity for tower {tower_id}: {severity.value}")
            return result
            
        except Exception as e:
            logging.error(f"Error classifying congestion severity for tower {tower_id}: {e}")
            return {
                "success": False,
                "message": f"Error classifying congestion severity: {str(e)}",
                "tower_id": tower_id
            }
    
    def configure_congestion_thresholds(self, low: float = 80.0, medium: float = 85.0, high: float = 95.0) -> Dict[str, Any]:
        """Configure congestion detection thresholds"""
        try:
            # Validate thresholds
            if not (0 <= low <= medium <= high <= 100):
                return {
                    "success": False,
                    "message": "Invalid threshold values. Must be: 0 <= low <= medium <= high <= 100",
                    "current_thresholds": self.severity_thresholds
                }
            
            self.severity_thresholds = {
                'low': low,
                'medium': medium,
                'high': high
            }
            
            # Update the main congestion threshold to the low threshold
            self.congestion_threshold = low
            
            logging.info(f"Updated congestion thresholds: LOW={low}%, MEDIUM={medium}%, HIGH={high}%")
            
            return {
                "success": True,
                "message": "Congestion thresholds updated successfully",
                "thresholds": self.severity_thresholds,
                "updated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error configuring congestion thresholds: {e}")
            return {
                "success": False,
                "message": f"Error configuring thresholds: {str(e)}",
                "current_thresholds": self.severity_thresholds
            }
    
    def detect_congestion_advanced(self, threshold: Optional[float] = None, include_predictions: bool = False) -> Dict[str, Any]:
        """Advanced congestion detection with configurable parameters"""
        try:
            if threshold is None:
                threshold = self.congestion_threshold
            
            congested_towers = self.db_manager.get_congested_towers(threshold)
            
            congestion_results = []
            new_events_created = 0
            severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            
            for tower_id, metrics in congested_towers:
                # Check if there's already an active congestion event
                active_events = self.db_manager.get_active_congestion_events()
                existing_event = next((e for e in active_events if e.tower_id == tower_id), None)
                
                # Calculate advanced severity based on multiple factors
                severity = self._calculate_advanced_severity(metrics)
                severity_counts[severity.value] += 1
                
                if not existing_event:
                    # Create new congestion event
                    tower = self.db_manager.get_tower_by_id(tower_id)
                    affected_area = tower.name if tower else "Unknown"
                    
                    event = CongestionEvent(
                        tower_id=tower_id,
                        severity=severity,
                        detected_at=datetime.now(),
                        metrics=metrics,
                        affected_area=affected_area
                    )
                    
                    event_id = self.db_manager.insert_congestion_event(event)
                    event.id = event_id
                    new_events_created += 1
                    
                    logging.warning(f"New congestion event created for tower {tower_id} "
                                  f"with severity {severity.value}")
                
                # Calculate congestion score (0-100)
                congestion_score = self._calculate_congestion_score(metrics)
                
                tower_result = {
                    "tower_id": tower_id,
                    "severity": severity.value,
                    "congestion_score": congestion_score,
                    "cpu_utilization": metrics.cpu_utilization,
                    "memory_usage": metrics.memory_usage,
                    "bandwidth_usage": metrics.bandwidth_usage,
                    "active_connections": metrics.active_connections,
                    "signal_strength": metrics.signal_strength,
                    "error_rate": metrics.error_rate,
                    "detected_at": metrics.timestamp.isoformat(),
                    "existing_event": existing_event is not None,
                    "trend_analysis": self._analyze_congestion_trend(tower_id)
                }
                
                # Add predictions if requested
                if include_predictions:
                    tower_result["predicted_escalation"] = self._predict_congestion_escalation(tower_id, metrics)
                
                congestion_results.append(tower_result)
            
            # Calculate network-wide congestion metrics
            network_health_score = self._calculate_network_health_score(congested_towers)
            
            result = {
                "success": True,
                "threshold_used": threshold,
                "total_congested_towers": len(congested_towers),
                "new_events_created": new_events_created,
                "severity_breakdown": severity_counts,
                "network_health_score": network_health_score,
                "congested_towers": congestion_results,
                "scan_timestamp": datetime.now().isoformat(),
                "thresholds_used": self.severity_thresholds
            }
            
            # Add network-wide recommendations
            result["network_recommendations"] = self._generate_network_recommendations(congested_towers, severity_counts)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in advanced congestion detection: {e}")
            return {
                "success": False,
                "message": f"Error in advanced congestion detection: {str(e)}",
                "threshold_used": threshold
            }
    
    def start_real_time_monitoring(self, interval_seconds: int = 30) -> Dict[str, Any]:
        """Start real-time congestion monitoring"""
        try:
            if self.monitoring_active:
                return {
                    "success": False,
                    "message": "Real-time monitoring is already active",
                    "monitoring_active": True
                }
            
            self.monitoring_active = True
            self.monitoring_interval = interval_seconds
            
            logging.info(f"Started real-time monitoring with {interval_seconds}s interval")
            
            return {
                "success": True,
                "message": f"Real-time monitoring started with {interval_seconds}s interval",
                "monitoring_active": True,
                "interval_seconds": interval_seconds,
                "started_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error starting real-time monitoring: {e}")
            return {
                "success": False,
                "message": f"Error starting monitoring: {str(e)}",
                "monitoring_active": False
            }
    
    def stop_real_time_monitoring(self) -> Dict[str, Any]:
        """Stop real-time congestion monitoring"""
        try:
            if not self.monitoring_active:
                return {
                    "success": False,
                    "message": "Real-time monitoring is not active",
                    "monitoring_active": False
                }
            
            self.monitoring_active = False
            
            logging.info("Stopped real-time monitoring")
            
            return {
                "success": True,
                "message": "Real-time monitoring stopped",
                "monitoring_active": False,
                "stopped_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error stopping real-time monitoring: {e}")
            return {
                "success": False,
                "message": f"Error stopping monitoring: {str(e)}",
                "monitoring_active": self.monitoring_active
            }
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and configuration"""
        try:
            active_events = self.db_manager.get_active_congestion_events()
            
            return {
                "success": True,
                "monitoring_active": self.monitoring_active,
                "monitoring_interval": getattr(self, 'monitoring_interval', None),
                "congestion_thresholds": self.severity_thresholds,
                "active_congestion_events": len(active_events),
                "status_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error getting monitoring status: {e}")
            return {
                "success": False,
                "message": f"Error getting monitoring status: {str(e)}"
            }
    
    def _calculate_advanced_severity(self, metrics: TowerMetrics) -> SeverityLevel:
        """Calculate congestion severity using advanced algorithms"""
        # Get maximum resource usage
        max_usage = max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage)
        
        # Apply weighted scoring (bandwidth is most critical for telecom)
        weighted_score = (
            metrics.cpu_utilization * 0.2 +
            metrics.memory_usage * 0.3 +
            metrics.bandwidth_usage * 0.5
        )
        
        # Factor in error rate (higher error rate increases severity)
        error_penalty = metrics.error_rate * 2  # Each 1% error adds 2 points
        adjusted_score = weighted_score + error_penalty
        
        # Determine severity based on thresholds
        if adjusted_score >= self.severity_thresholds['high']:
            return SeverityLevel.HIGH
        elif adjusted_score >= self.severity_thresholds['medium']:
            return SeverityLevel.MEDIUM
        elif adjusted_score >= self.severity_thresholds['low']:
            return SeverityLevel.LOW
        else:
            return SeverityLevel.LOW
    
    def _calculate_congestion_score(self, metrics: TowerMetrics) -> float:
        """Calculate a congestion score from 0-100"""
        # Weighted average of resource utilization
        score = (
            metrics.cpu_utilization * 0.2 +
            metrics.memory_usage * 0.3 +
            metrics.bandwidth_usage * 0.5
        )
        
        # Add penalty for high error rate
        error_penalty = min(metrics.error_rate * 5, 20)  # Cap at 20 points
        
        # Add penalty for poor signal strength (below -80 dBm is concerning)
        if metrics.signal_strength < -80:
            signal_penalty = abs(metrics.signal_strength + 80) * 0.5
        else:
            signal_penalty = 0
        
        final_score = min(score + error_penalty + signal_penalty, 100)
        return round(final_score, 2)
    
    def _analyze_congestion_trend(self, tower_id: str) -> Dict[str, Any]:
        """Analyze congestion trend for a tower"""
        try:
            # Get recent metrics (last 2 hours)
            recent_metrics = self.db_manager.get_tower_metrics_history(tower_id, hours=2)
            
            if len(recent_metrics) < 2:
                return {"trend": "insufficient_data", "confidence": 0.0}
            
            # Calculate trend for each metric
            cpu_trend = self._calculate_metric_trend([m.cpu_utilization for m in recent_metrics])
            memory_trend = self._calculate_metric_trend([m.memory_usage for m in recent_metrics])
            bandwidth_trend = self._calculate_metric_trend([m.bandwidth_usage for m in recent_metrics])
            
            # Overall trend (weighted average)
            overall_trend = (cpu_trend * 0.2 + memory_trend * 0.3 + bandwidth_trend * 0.5)
            
            # Determine trend direction
            if overall_trend > 2:
                trend_direction = "increasing"
            elif overall_trend < -2:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            return {
                "trend": trend_direction,
                "trend_score": round(overall_trend, 2),
                "confidence": min(len(recent_metrics) / 10.0, 1.0),  # More data = higher confidence
                "cpu_trend": round(cpu_trend, 2),
                "memory_trend": round(memory_trend, 2),
                "bandwidth_trend": round(bandwidth_trend, 2)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing congestion trend for {tower_id}: {e}")
            return {"trend": "error", "confidence": 0.0, "error": str(e)}
    
    def _calculate_metric_trend(self, values: List[float]) -> float:
        """Calculate trend for a list of metric values (positive = increasing)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Calculate slope (trend)
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    def _predict_congestion_escalation(self, tower_id: str, current_metrics: TowerMetrics) -> Dict[str, Any]:
        """Predict if congestion will escalate in the next hour"""
        try:
            trend_analysis = self._analyze_congestion_trend(tower_id)
            
            if trend_analysis["trend"] == "increasing" and trend_analysis["confidence"] > 0.5:
                # Predict escalation based on current level and trend
                current_max = max(current_metrics.cpu_utilization, 
                                current_metrics.memory_usage, 
                                current_metrics.bandwidth_usage)
                
                predicted_increase = trend_analysis["trend_score"] * 6  # 6 intervals in 1 hour (10min each)
                predicted_level = current_max + predicted_increase
                
                escalation_risk = "high" if predicted_level > 95 else "medium" if predicted_level > 85 else "low"
                
                return {
                    "escalation_risk": escalation_risk,
                    "predicted_level": min(predicted_level, 100),
                    "confidence": trend_analysis["confidence"],
                    "time_horizon_minutes": 60
                }
            else:
                return {
                    "escalation_risk": "low",
                    "predicted_level": max(current_metrics.cpu_utilization, 
                                         current_metrics.memory_usage, 
                                         current_metrics.bandwidth_usage),
                    "confidence": trend_analysis["confidence"],
                    "time_horizon_minutes": 60
                }
                
        except Exception as e:
            logging.error(f"Error predicting congestion escalation for {tower_id}: {e}")
            return {
                "escalation_risk": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_network_health_score(self, congested_towers: List[Tuple[str, TowerMetrics]]) -> float:
        """Calculate overall network health score (0-100)"""
        try:
            all_towers = self.db_manager.get_active_towers()
            if not all_towers:
                return 0.0
            
            total_towers = len(all_towers)
            congested_count = len(congested_towers)
            
            # Base score starts at 100 and decreases with congestion
            base_score = 100.0
            
            # Penalty for congested towers (more severe = higher penalty)
            congestion_penalty = 0
            for tower_id, metrics in congested_towers:
                severity = self._calculate_advanced_severity(metrics)
                if severity == SeverityLevel.HIGH:
                    congestion_penalty += 15
                elif severity == SeverityLevel.MEDIUM:
                    congestion_penalty += 8
                else:  # LOW
                    congestion_penalty += 3
            
            # Additional penalty for high percentage of congested towers
            percentage_penalty = (congested_count / total_towers) * 30
            
            final_score = max(base_score - congestion_penalty - percentage_penalty, 0.0)
            return round(final_score, 2)
            
        except Exception as e:
            logging.error(f"Error calculating network health score: {e}")
            return 0.0
    
    def _generate_network_recommendations(self, congested_towers: List[Tuple[str, TowerMetrics]], 
                                        severity_counts: Dict[str, int]) -> List[str]:
        """Generate network-wide recommendations based on congestion analysis"""
        recommendations = []
        
        total_congested = sum(severity_counts.values())
        
        if severity_counts["HIGH"] > 0:
            recommendations.append(f"URGENT: {severity_counts['HIGH']} towers with HIGH congestion - immediate intervention required")
            recommendations.append("Activate emergency load balancing procedures")
            recommendations.append("Consider traffic redirection to neighboring towers")
            
        if severity_counts["MEDIUM"] > 2:
            recommendations.append(f"WARNING: {severity_counts['MEDIUM']} towers with MEDIUM congestion")
            recommendations.append("Prepare proactive spectrum reallocation")
            recommendations.append("Monitor for potential escalation to HIGH severity")
            
        if total_congested > 5:
            recommendations.append("Network-wide congestion detected - review capacity planning")
            recommendations.append("Consider infrastructure upgrades for high-traffic areas")
            
        if severity_counts["LOW"] > 0 and not recommendations:
            recommendations.append(f"Monitor {severity_counts['LOW']} towers with LOW congestion")
            recommendations.append("Continue regular monitoring - no immediate action required")
            
        if not recommendations:
            recommendations.append("Network operating within normal parameters")
            
        return recommendations
    
    def get_load_history_advanced(self, tower_id: str, hours: int = 24, include_analysis: bool = True) -> Dict[str, Any]:
        """Get advanced historical load data analysis for a tower"""
        try:
            metrics_history = self.db_manager.get_tower_metrics_history(tower_id, hours)
            
            if not metrics_history:
                return {
                    "success": False,
                    "message": f"No historical data found for tower {tower_id}",
                    "tower_id": tower_id,
                    "hours_requested": hours
                }
            
            # Basic statistics
            cpu_values = [m.cpu_utilization for m in metrics_history]
            memory_values = [m.memory_usage for m in metrics_history]
            bandwidth_values = [m.bandwidth_usage for m in metrics_history]
            connection_values = [m.active_connections for m in metrics_history]
            error_values = [m.error_rate for m in metrics_history]
            signal_values = [m.signal_strength for m in metrics_history]
            
            # Count congestion periods
            congestion_periods = len([m for m in metrics_history if m.is_congested])
            
            # Find peak usage periods
            peak_cpu_metric = max(metrics_history, key=lambda m: m.cpu_utilization)
            peak_bandwidth_metric = max(metrics_history, key=lambda m: m.bandwidth_usage)
            peak_connections_metric = max(metrics_history, key=lambda m: m.active_connections)
            
            result = {
                "success": True,
                "tower_id": tower_id,
                "period_hours": hours,
                "total_measurements": len(metrics_history),
                "data_quality": {
                    "completeness": self._calculate_data_completeness(metrics_history, hours),
                    "consistency": self._calculate_data_consistency(metrics_history),
                    "latest_measurement": metrics_history[0].timestamp.isoformat() if metrics_history else None
                },
                "statistics": {
                    "cpu_utilization": self._calculate_metric_statistics(cpu_values, peak_cpu_metric.timestamp),
                    "memory_usage": self._calculate_metric_statistics(memory_values),
                    "bandwidth_usage": self._calculate_metric_statistics(bandwidth_values, peak_bandwidth_metric.timestamp),
                    "active_connections": self._calculate_metric_statistics(connection_values, peak_connections_metric.timestamp),
                    "error_rate": self._calculate_metric_statistics(error_values),
                    "signal_strength": self._calculate_metric_statistics(signal_values)
                },
                "congestion_analysis": {
                    "total_congestion_periods": congestion_periods,
                    "congestion_percentage": round((congestion_periods / len(metrics_history)) * 100, 2),
                    "congestion_frequency": self._analyze_congestion_frequency(metrics_history),
                    "congestion_duration_analysis": self._analyze_congestion_durations(metrics_history)
                },
                "recent_metrics": [m.to_dict() for m in metrics_history[:5]],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Add advanced analysis if requested
            if include_analysis:
                result["advanced_analysis"] = {
                    "trend_analysis": self._analyze_congestion_trend(tower_id),
                    "pattern_recognition": self._recognize_usage_patterns(metrics_history),
                    "performance_insights": self._generate_performance_insights(metrics_history),
                    "capacity_analysis": self._analyze_capacity_utilization(metrics_history),
                    "anomaly_detection": self._detect_anomalies(metrics_history)
                }
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting advanced load history for tower {tower_id}: {e}")
            return {
                "success": False,
                "message": f"Error getting advanced load history: {str(e)}",
                "tower_id": tower_id,
                "hours_requested": hours
            }
    
    def generate_performance_report(self, tower_ids: List[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report for towers"""
        try:
            if tower_ids is None:
                towers = self.db_manager.get_active_towers()
                tower_ids = [t.id for t in towers]
            
            if not tower_ids:
                return {
                    "success": False,
                    "message": "No towers specified or found for report generation"
                }
            
            tower_reports = []
            network_summary = {
                "total_towers": len(tower_ids),
                "reporting_period_hours": hours,
                "congested_towers": 0,
                "high_severity_towers": 0,
                "average_network_utilization": 0.0,
                "total_active_connections": 0
            }
            
            total_utilization = 0
            total_connections = 0
            
            for tower_id in tower_ids:
                tower_report = self.get_load_history_advanced(tower_id, hours, include_analysis=True)
                
                if tower_report["success"]:
                    tower_reports.append(tower_report)
                    
                    # Update network summary
                    stats = tower_report["statistics"]
                    if tower_report["congestion_analysis"]["congestion_percentage"] > 0:
                        network_summary["congested_towers"] += 1
                    
                    # Check for high severity (>90% average utilization)
                    avg_bandwidth = stats["bandwidth_usage"]["average"]
                    if avg_bandwidth > 90:
                        network_summary["high_severity_towers"] += 1
                    
                    total_utilization += avg_bandwidth
                    total_connections += stats["active_connections"]["average"]
            
            # Calculate network averages
            if tower_reports:
                network_summary["average_network_utilization"] = round(total_utilization / len(tower_reports), 2)
                network_summary["total_active_connections"] = round(total_connections, 0)
            
            # Generate network-wide recommendations
            network_recommendations = self._generate_performance_recommendations(tower_reports, network_summary)
            
            return {
                "success": True,
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "reporting_period_hours": hours,
                    "towers_analyzed": len(tower_reports),
                    "towers_requested": len(tower_ids)
                },
                "network_summary": network_summary,
                "tower_reports": tower_reports,
                "network_recommendations": network_recommendations,
                "report_insights": self._generate_report_insights(tower_reports, network_summary)
            }
            
        except Exception as e:
            logging.error(f"Error generating performance report: {e}")
            return {
                "success": False,
                "message": f"Error generating performance report: {str(e)}"
            }
    
    def analyze_network_patterns(self, days: int = 7) -> Dict[str, Any]:
        """Analyze network-wide usage patterns over multiple days"""
        try:
            hours = days * 24
            towers = self.db_manager.get_active_towers()
            
            if not towers:
                return {
                    "success": False,
                    "message": "No active towers found for pattern analysis"
                }
            
            pattern_analysis = {
                "analysis_period_days": days,
                "towers_analyzed": len(towers),
                "hourly_patterns": {},
                "daily_patterns": {},
                "weekly_patterns": {},
                "congestion_patterns": {},
                "capacity_trends": {}
            }
            
            all_metrics = []
            
            # Collect metrics from all towers
            for tower in towers:
                tower_metrics = self.db_manager.get_tower_metrics_history(tower.id, hours)
                for metric in tower_metrics:
                    all_metrics.append({
                        "tower_id": tower.id,
                        "timestamp": metric.timestamp,
                        "hour": metric.timestamp.hour,
                        "day_of_week": metric.timestamp.weekday(),
                        "cpu_utilization": metric.cpu_utilization,
                        "memory_usage": metric.memory_usage,
                        "bandwidth_usage": metric.bandwidth_usage,
                        "active_connections": metric.active_connections,
                        "is_congested": metric.is_congested
                    })
            
            if not all_metrics:
                return {
                    "success": False,
                    "message": "No metrics data found for pattern analysis"
                }
            
            # Analyze hourly patterns (0-23)
            pattern_analysis["hourly_patterns"] = self._analyze_hourly_patterns(all_metrics)
            
            # Analyze daily patterns (Monday=0 to Sunday=6)
            pattern_analysis["daily_patterns"] = self._analyze_daily_patterns(all_metrics)
            
            # Analyze congestion patterns
            pattern_analysis["congestion_patterns"] = self._analyze_congestion_patterns(all_metrics)
            
            # Analyze capacity trends
            pattern_analysis["capacity_trends"] = self._analyze_capacity_trends(all_metrics)
            
            return {
                "success": True,
                "pattern_analysis": pattern_analysis,
                "analysis_timestamp": datetime.now().isoformat(),
                "insights": self._generate_pattern_insights(pattern_analysis)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing network patterns: {e}")
            return {
                "success": False,
                "message": f"Error analyzing network patterns: {str(e)}"
            }
    
    def _calculate_data_completeness(self, metrics_history: List[TowerMetrics], hours: int) -> float:
        """Calculate data completeness percentage"""
        expected_measurements = hours * 6  # Assuming 10-minute intervals
        actual_measurements = len(metrics_history)
        return min(round((actual_measurements / expected_measurements) * 100, 2), 100.0)
    
    def _calculate_data_consistency(self, metrics_history: List[TowerMetrics]) -> Dict[str, Any]:
        """Calculate data consistency metrics"""
        if len(metrics_history) < 2:
            return {"score": 100.0, "issues": []}
        
        issues = []
        
        # Check for missing timestamps (gaps > 15 minutes)
        time_gaps = []
        for i in range(1, len(metrics_history)):
            gap = (metrics_history[i-1].timestamp - metrics_history[i].timestamp).total_seconds() / 60
            if gap > 15:  # More than 15 minutes
                time_gaps.append(gap)
        
        if time_gaps:
            issues.append(f"Found {len(time_gaps)} time gaps > 15 minutes")
        
        # Check for unrealistic value jumps
        for i in range(1, len(metrics_history)):
            prev = metrics_history[i-1]
            curr = metrics_history[i]
            
            if abs(prev.cpu_utilization - curr.cpu_utilization) > 30:
                issues.append("Large CPU utilization jumps detected")
                break
        
        consistency_score = max(100 - len(issues) * 20, 0)
        
        return {
            "score": consistency_score,
            "issues": issues,
            "time_gaps": len(time_gaps),
            "average_gap_minutes": round(sum(time_gaps) / len(time_gaps), 2) if time_gaps else 0
        }
    
    def _calculate_metric_statistics(self, values: List[float], peak_timestamp: datetime = None) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a metric"""
        if not values:
            return {"error": "No data available"}
        
        sorted_values = sorted(values)
        n = len(values)
        
        stats = {
            "count": n,
            "average": round(sum(values) / n, 2),
            "maximum": max(values),
            "minimum": min(values),
            "median": sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2,
            "percentile_95": sorted_values[int(n * 0.95)],
            "percentile_75": sorted_values[int(n * 0.75)],
            "percentile_25": sorted_values[int(n * 0.25)],
            "standard_deviation": self._calculate_std_dev(values),
            "variance": round(sum((x - sum(values) / n) ** 2 for x in values) / n, 2)
        }
        
        if peak_timestamp:
            stats["peak_timestamp"] = peak_timestamp.isoformat()
        
        return stats
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return round(variance ** 0.5, 2)
    
    def _analyze_congestion_frequency(self, metrics_history: List[TowerMetrics]) -> Dict[str, Any]:
        """Analyze congestion frequency patterns"""
        if not metrics_history:
            return {"error": "No data available"}
        
        congestion_events = []
        in_congestion = False
        congestion_start = None
        
        for metric in reversed(metrics_history):  # Process chronologically
            if metric.is_congested and not in_congestion:
                # Start of congestion period
                in_congestion = True
                congestion_start = metric.timestamp
            elif not metric.is_congested and in_congestion:
                # End of congestion period
                in_congestion = False
                if congestion_start:
                    duration = (metric.timestamp - congestion_start).total_seconds() / 60
                    congestion_events.append({
                        "start": congestion_start.isoformat(),
                        "end": metric.timestamp.isoformat(),
                        "duration_minutes": duration
                    })
        
        if not congestion_events:
            return {
                "total_events": 0,
                "average_duration_minutes": 0,
                "longest_event_minutes": 0,
                "shortest_event_minutes": 0
            }
        
        durations = [event["duration_minutes"] for event in congestion_events]
        
        return {
            "total_events": len(congestion_events),
            "average_duration_minutes": round(sum(durations) / len(durations), 2),
            "longest_event_minutes": max(durations),
            "shortest_event_minutes": min(durations),
            "events": congestion_events[-5:]  # Last 5 events
        }
    
    def _analyze_congestion_durations(self, metrics_history: List[TowerMetrics]) -> Dict[str, Any]:
        """Analyze congestion duration patterns"""
        frequency_analysis = self._analyze_congestion_frequency(metrics_history)
        
        if frequency_analysis.get("total_events", 0) == 0:
            return {"pattern": "no_congestion", "analysis": "No congestion events detected"}
        
        avg_duration = frequency_analysis["average_duration_minutes"]
        
        if avg_duration < 30:
            pattern = "short_bursts"
            analysis = "Congestion occurs in short bursts, likely due to temporary traffic spikes"
        elif avg_duration < 120:
            pattern = "moderate_duration"
            analysis = "Moderate congestion periods, may indicate capacity constraints during peak times"
        else:
            pattern = "prolonged_congestion"
            analysis = "Prolonged congestion periods, indicates serious capacity issues requiring intervention"
        
        return {
            "pattern": pattern,
            "analysis": analysis,
            "average_duration_minutes": avg_duration,
            "total_events": frequency_analysis["total_events"]
        }
    
    def _recognize_usage_patterns(self, metrics_history: List[TowerMetrics]) -> Dict[str, Any]:
        """Recognize usage patterns in historical data"""
        if len(metrics_history) < 24:  # Need at least 24 data points
            return {"pattern": "insufficient_data", "confidence": 0.0}
        
        # Group by hour of day
        hourly_usage = {}
        for metric in metrics_history:
            hour = metric.timestamp.hour
            if hour not in hourly_usage:
                hourly_usage[hour] = []
            hourly_usage[hour].append(metric.bandwidth_usage)
        
        # Calculate average usage by hour
        hourly_averages = {}
        for hour, usages in hourly_usage.items():
            hourly_averages[hour] = sum(usages) / len(usages)
        
        # Identify peak hours (top 25% of hours)
        sorted_hours = sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)
        peak_hours = [hour for hour, usage in sorted_hours[:6]]  # Top 6 hours
        
        # Identify pattern type
        if all(9 <= hour <= 17 for hour in peak_hours):
            pattern_type = "business_hours"
            description = "Peak usage during business hours (9 AM - 5 PM)"
        elif all(18 <= hour <= 23 or 0 <= hour <= 2 for hour in peak_hours):
            pattern_type = "evening_peak"
            description = "Peak usage during evening hours (6 PM - 2 AM)"
        elif len(set(peak_hours)) > 12:
            pattern_type = "distributed"
            description = "Usage distributed throughout the day"
        else:
            pattern_type = "mixed"
            description = "Mixed usage pattern with multiple peak periods"
        
        return {
            "pattern": pattern_type,
            "description": description,
            "peak_hours": sorted(peak_hours),
            "hourly_averages": hourly_averages,
            "confidence": min(len(metrics_history) / 168.0, 1.0)  # Higher confidence with more data (1 week = 168 hours)
        }
    
    def _generate_performance_insights(self, metrics_history: List[TowerMetrics]) -> List[str]:
        """Generate performance insights from historical data"""
        insights = []
        
        if not metrics_history:
            return ["No data available for insights generation"]
        
        # Analyze recent trend
        recent_metrics = metrics_history[:12]  # Last 2 hours
        if len(recent_metrics) >= 6:
            recent_bandwidth = [m.bandwidth_usage for m in recent_metrics]
            trend = self._calculate_metric_trend(recent_bandwidth)
            
            if trend > 5:
                insights.append("⚠️ Bandwidth usage is rapidly increasing - monitor for potential congestion")
            elif trend < -5:
                insights.append("✅ Bandwidth usage is decreasing - congestion risk is reducing")
        
        # Analyze error rates
        error_rates = [m.error_rate for m in metrics_history]
        avg_error_rate = sum(error_rates) / len(error_rates)
        
        if avg_error_rate > 1.0:
            insights.append(f"🔴 High average error rate ({avg_error_rate:.2f}%) indicates network quality issues")
        elif avg_error_rate > 0.5:
            insights.append(f"🟡 Moderate error rate ({avg_error_rate:.2f}%) - monitor network quality")
        else:
            insights.append(f"✅ Low error rate ({avg_error_rate:.2f}%) indicates good network quality")
        
        # Analyze signal strength
        signal_strengths = [m.signal_strength for m in metrics_history]
        avg_signal = sum(signal_strengths) / len(signal_strengths)
        
        if avg_signal < -80:
            insights.append(f"📶 Poor signal strength ({avg_signal:.1f} dBm) may impact performance")
        elif avg_signal < -70:
            insights.append(f"📶 Moderate signal strength ({avg_signal:.1f} dBm)")
        else:
            insights.append(f"📶 Good signal strength ({avg_signal:.1f} dBm)")
        
        # Analyze capacity utilization
        bandwidth_values = [m.bandwidth_usage for m in metrics_history]
        max_bandwidth = max(bandwidth_values)
        avg_bandwidth = sum(bandwidth_values) / len(bandwidth_values)
        
        if max_bandwidth > 95:
            insights.append("🚨 Tower reached critical capacity (>95%) - immediate attention required")
        elif avg_bandwidth > 80:
            insights.append("⚠️ High average utilization - consider capacity expansion")
        elif avg_bandwidth < 30:
            insights.append("💡 Low utilization - tower may be underutilized or in low-demand area")
        
        return insights
    
    def _analyze_capacity_utilization(self, metrics_history: List[TowerMetrics]) -> Dict[str, Any]:
        """Analyze capacity utilization patterns"""
        if not metrics_history:
            return {"error": "No data available"}
        
        bandwidth_values = [m.bandwidth_usage for m in metrics_history]
        
        # Calculate utilization bands
        utilization_bands = {
            "low": len([v for v in bandwidth_values if v < 50]),
            "medium": len([v for v in bandwidth_values if 50 <= v < 80]),
            "high": len([v for v in bandwidth_values if 80 <= v < 95]),
            "critical": len([v for v in bandwidth_values if v >= 95])
        }
        
        total_measurements = len(bandwidth_values)
        utilization_percentages = {
            band: round((count / total_measurements) * 100, 2)
            for band, count in utilization_bands.items()
        }
        
        # Determine capacity status
        if utilization_percentages["critical"] > 10:
            capacity_status = "critical"
            recommendation = "Immediate capacity expansion required"
        elif utilization_percentages["high"] > 30:
            capacity_status = "strained"
            recommendation = "Consider capacity expansion planning"
        elif utilization_percentages["medium"] > 60:
            capacity_status = "optimal"
            recommendation = "Capacity utilization is within optimal range"
        else:
            capacity_status = "underutilized"
            recommendation = "Tower appears underutilized - review coverage area"
        
        return {
            "capacity_status": capacity_status,
            "recommendation": recommendation,
            "utilization_bands": utilization_bands,
            "utilization_percentages": utilization_percentages,
            "peak_utilization": max(bandwidth_values),
            "average_utilization": round(sum(bandwidth_values) / len(bandwidth_values), 2)
        }
    
    def _detect_anomalies(self, metrics_history: List[TowerMetrics]) -> Dict[str, Any]:
        """Detect anomalies in tower metrics"""
        if len(metrics_history) < 10:
            return {"anomalies_detected": 0, "details": "Insufficient data for anomaly detection"}
        
        anomalies = []
        
        # Calculate baseline statistics
        bandwidth_values = [m.bandwidth_usage for m in metrics_history]
        cpu_values = [m.cpu_utilization for m in metrics_history]
        error_values = [m.error_rate for m in metrics_history]
        
        bandwidth_mean = sum(bandwidth_values) / len(bandwidth_values)
        bandwidth_std = self._calculate_std_dev(bandwidth_values)
        
        error_mean = sum(error_values) / len(error_values)
        error_std = self._calculate_std_dev(error_values)
        
        # Detect anomalies (values > 2 standard deviations from mean)
        for i, metric in enumerate(metrics_history):
            # Bandwidth anomalies
            if abs(metric.bandwidth_usage - bandwidth_mean) > 2 * bandwidth_std:
                anomalies.append({
                    "type": "bandwidth_anomaly",
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.bandwidth_usage,
                    "expected_range": f"{bandwidth_mean - 2*bandwidth_std:.1f} - {bandwidth_mean + 2*bandwidth_std:.1f}",
                    "severity": "high" if abs(metric.bandwidth_usage - bandwidth_mean) > 3 * bandwidth_std else "medium"
                })
            
            # Error rate spikes
            if metric.error_rate > error_mean + 2 * error_std and metric.error_rate > 1.0:
                anomalies.append({
                    "type": "error_spike",
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.error_rate,
                    "expected_range": f"< {error_mean + 2*error_std:.2f}",
                    "severity": "high" if metric.error_rate > 5.0 else "medium"
                })
            
            # Signal strength drops
            if metric.signal_strength < -85:  # Very poor signal
                anomalies.append({
                    "type": "signal_degradation",
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.signal_strength,
                    "expected_range": "> -80 dBm",
                    "severity": "medium"
                })
        
        return {
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies[-10:],  # Last 10 anomalies
            "anomaly_types": list(set(a["type"] for a in anomalies)),
            "high_severity_count": len([a for a in anomalies if a["severity"] == "high"])
        }
    
    def _analyze_hourly_patterns(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze usage patterns by hour of day"""
        hourly_data = {}
        
        for metric in all_metrics:
            hour = metric["hour"]
            if hour not in hourly_data:
                hourly_data[hour] = {
                    "bandwidth_usage": [],
                    "cpu_utilization": [],
                    "active_connections": [],
                    "congestion_count": 0,
                    "total_measurements": 0
                }
            
            hourly_data[hour]["bandwidth_usage"].append(metric["bandwidth_usage"])
            hourly_data[hour]["cpu_utilization"].append(metric["cpu_utilization"])
            hourly_data[hour]["active_connections"].append(metric["active_connections"])
            hourly_data[hour]["total_measurements"] += 1
            
            if metric["is_congested"]:
                hourly_data[hour]["congestion_count"] += 1
        
        # Calculate averages and patterns
        hourly_patterns = {}
        for hour in range(24):
            if hour in hourly_data:
                data = hourly_data[hour]
                hourly_patterns[hour] = {
                    "average_bandwidth": round(sum(data["bandwidth_usage"]) / len(data["bandwidth_usage"]), 2),
                    "average_cpu": round(sum(data["cpu_utilization"]) / len(data["cpu_utilization"]), 2),
                    "average_connections": round(sum(data["active_connections"]) / len(data["active_connections"]), 0),
                    "congestion_rate": round((data["congestion_count"] / data["total_measurements"]) * 100, 2),
                    "measurements": data["total_measurements"]
                }
            else:
                hourly_patterns[hour] = {
                    "average_bandwidth": 0,
                    "average_cpu": 0,
                    "average_connections": 0,
                    "congestion_rate": 0,
                    "measurements": 0
                }
        
        # Identify peak hours
        peak_hours = sorted(hourly_patterns.items(), key=lambda x: x[1]["average_bandwidth"], reverse=True)[:6]
        
        return {
            "hourly_averages": hourly_patterns,
            "peak_hours": [{"hour": hour, "usage": data["average_bandwidth"]} for hour, data in peak_hours],
            "lowest_usage_hour": min(hourly_patterns.items(), key=lambda x: x[1]["average_bandwidth"])[0],
            "highest_usage_hour": max(hourly_patterns.items(), key=lambda x: x[1]["average_bandwidth"])[0]
        }
    
    def _analyze_daily_patterns(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze usage patterns by day of week"""
        daily_data = {}
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for metric in all_metrics:
            day = metric["day_of_week"]
            if day not in daily_data:
                daily_data[day] = {
                    "bandwidth_usage": [],
                    "congestion_count": 0,
                    "total_measurements": 0
                }
            
            daily_data[day]["bandwidth_usage"].append(metric["bandwidth_usage"])
            daily_data[day]["total_measurements"] += 1
            
            if metric["is_congested"]:
                daily_data[day]["congestion_count"] += 1
        
        # Calculate daily patterns
        daily_patterns = {}
        for day in range(7):
            day_name = day_names[day]
            if day in daily_data:
                data = daily_data[day]
                daily_patterns[day_name] = {
                    "average_bandwidth": round(sum(data["bandwidth_usage"]) / len(data["bandwidth_usage"]), 2),
                    "congestion_rate": round((data["congestion_count"] / data["total_measurements"]) * 100, 2),
                    "measurements": data["total_measurements"]
                }
            else:
                daily_patterns[day_name] = {
                    "average_bandwidth": 0,
                    "congestion_rate": 0,
                    "measurements": 0
                }
        
        # Identify patterns
        weekday_avg = sum(daily_patterns[day]["average_bandwidth"] for day in day_names[:5]) / 5
        weekend_avg = sum(daily_patterns[day]["average_bandwidth"] for day in day_names[5:]) / 2
        
        pattern_type = "weekday_heavy" if weekday_avg > weekend_avg * 1.2 else "weekend_heavy" if weekend_avg > weekday_avg * 1.2 else "balanced"
        
        return {
            "daily_averages": daily_patterns,
            "weekday_average": round(weekday_avg, 2),
            "weekend_average": round(weekend_avg, 2),
            "pattern_type": pattern_type,
            "busiest_day": max(daily_patterns.items(), key=lambda x: x[1]["average_bandwidth"])[0],
            "quietest_day": min(daily_patterns.items(), key=lambda x: x[1]["average_bandwidth"])[0]
        }
    
    def _analyze_congestion_patterns(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze when congestion typically occurs"""
        congested_metrics = [m for m in all_metrics if m["is_congested"]]
        
        if not congested_metrics:
            return {
                "total_congestion_events": 0,
                "congestion_by_hour": {},
                "congestion_by_day": {},
                "pattern": "no_congestion"
            }
        
        # Analyze congestion by hour
        congestion_by_hour = {}
        for metric in congested_metrics:
            hour = metric["hour"]
            congestion_by_hour[hour] = congestion_by_hour.get(hour, 0) + 1
        
        # Analyze congestion by day
        congestion_by_day = {}
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for metric in congested_metrics:
            day_name = day_names[metric["day_of_week"]]
            congestion_by_day[day_name] = congestion_by_day.get(day_name, 0) + 1
        
        # Identify peak congestion times
        peak_congestion_hour = max(congestion_by_hour.items(), key=lambda x: x[1])[0] if congestion_by_hour else None
        peak_congestion_day = max(congestion_by_day.items(), key=lambda x: x[1])[0] if congestion_by_day else None
        
        return {
            "total_congestion_events": len(congested_metrics),
            "congestion_by_hour": congestion_by_hour,
            "congestion_by_day": congestion_by_day,
            "peak_congestion_hour": peak_congestion_hour,
            "peak_congestion_day": peak_congestion_day,
            "congestion_rate": round((len(congested_metrics) / len(all_metrics)) * 100, 2)
        }
    
    def _analyze_capacity_trends(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze capacity utilization trends over time"""
        if len(all_metrics) < 24:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Sort by timestamp
        sorted_metrics = sorted(all_metrics, key=lambda x: x["timestamp"])
        
        # Calculate trend over time
        bandwidth_values = [m["bandwidth_usage"] for m in sorted_metrics]
        trend_slope = self._calculate_metric_trend(bandwidth_values)
        
        # Analyze trend direction
        if trend_slope > 1:
            trend_direction = "increasing"
            trend_description = "Network capacity utilization is increasing over time"
        elif trend_slope < -1:
            trend_direction = "decreasing"
            trend_description = "Network capacity utilization is decreasing over time"
        else:
            trend_direction = "stable"
            trend_description = "Network capacity utilization is stable"
        
        # Calculate growth rate (if increasing)
        growth_rate = None
        if trend_direction == "increasing":
            first_week_avg = sum(bandwidth_values[:168]) / min(len(bandwidth_values), 168)  # First week
            last_week_avg = sum(bandwidth_values[-168:]) / min(len(bandwidth_values[-168:]), 168)  # Last week
            if first_week_avg > 0:
                growth_rate = round(((last_week_avg - first_week_avg) / first_week_avg) * 100, 2)
        
        return {
            "trend": trend_direction,
            "trend_slope": round(trend_slope, 4),
            "description": trend_description,
            "growth_rate_percent": growth_rate,
            "confidence": min(len(all_metrics) / 168.0, 1.0)  # Higher confidence with more data
        }
    
    def _generate_performance_recommendations(self, tower_reports: List[Dict], network_summary: Dict) -> List[str]:
        """Generate performance recommendations based on tower reports"""
        recommendations = []
        
        # Network-wide recommendations
        congestion_rate = (network_summary["congested_towers"] / network_summary["total_towers"]) * 100
        
        if congestion_rate > 50:
            recommendations.append("🚨 CRITICAL: Over 50% of towers experiencing congestion - network-wide capacity expansion needed")
        elif congestion_rate > 25:
            recommendations.append("⚠️ WARNING: High congestion rate across network - review capacity planning")
        elif congestion_rate > 10:
            recommendations.append("📊 MONITOR: Moderate congestion levels - continue monitoring trends")
        
        # High severity towers
        if network_summary["high_severity_towers"] > 0:
            recommendations.append(f"🔴 {network_summary['high_severity_towers']} towers with >90% utilization require immediate attention")
        
        # Average utilization recommendations
        avg_util = network_summary["average_network_utilization"]
        if avg_util > 80:
            recommendations.append("📈 Network approaching capacity limits - plan infrastructure expansion")
        elif avg_util < 30:
            recommendations.append("💡 Network underutilized - review coverage optimization opportunities")
        
        # Tower-specific recommendations
        high_error_towers = []
        poor_signal_towers = []
        
        for report in tower_reports:
            if report["success"]:
                tower_id = report["tower_id"]
                stats = report["statistics"]
                
                if stats["error_rate"]["average"] > 1.0:
                    high_error_towers.append(tower_id)
                
                if stats["signal_strength"]["average"] < -80:
                    poor_signal_towers.append(tower_id)
        
        if high_error_towers:
            recommendations.append(f"🔧 Towers with high error rates need maintenance: {', '.join(high_error_towers[:3])}")
        
        if poor_signal_towers:
            recommendations.append(f"📶 Towers with poor signal strength: {', '.join(poor_signal_towers[:3])}")
        
        return recommendations
    
    def _generate_report_insights(self, tower_reports: List[Dict], network_summary: Dict) -> List[str]:
        """Generate insights from performance report"""
        insights = []
        
        if not tower_reports:
            return ["No tower data available for insights"]
        
        # Network health insight
        health_score = 100 - (network_summary["congested_towers"] / network_summary["total_towers"]) * 50
        if health_score > 80:
            insights.append(f"✅ Network health is good ({health_score:.0f}/100)")
        elif health_score > 60:
            insights.append(f"⚠️ Network health is moderate ({health_score:.0f}/100)")
        else:
            insights.append(f"🚨 Network health is poor ({health_score:.0f}/100)")
        
        # Capacity insight
        avg_util = network_summary["average_network_utilization"]
        if avg_util > 85:
            insights.append("🔴 Network operating near capacity - expansion planning critical")
        elif avg_util > 70:
            insights.append("🟡 Network utilization is high - monitor growth trends")
        else:
            insights.append("🟢 Network has adequate capacity headroom")
        
        # Pattern insights from individual towers
        pattern_types = []
        for report in tower_reports:
            if report["success"] and "advanced_analysis" in report:
                pattern = report["advanced_analysis"]["pattern_recognition"]["pattern"]
                pattern_types.append(pattern)
        
        if pattern_types:
            most_common_pattern = max(set(pattern_types), key=pattern_types.count)
            insights.append(f"📊 Most common usage pattern: {most_common_pattern}")
        
        return insights
    
    def _generate_pattern_insights(self, pattern_analysis: Dict) -> List[str]:
        """Generate insights from pattern analysis"""
        insights = []
        
        # Hourly pattern insights
        hourly = pattern_analysis.get("hourly_patterns", {})
        if "peak_hours" in hourly:
            peak_hours = [str(h["hour"]) for h in hourly["peak_hours"][:3]]
            insights.append(f"📈 Peak usage hours: {', '.join(peak_hours)}")
        
        # Daily pattern insights
        daily = pattern_analysis.get("daily_patterns", {})
        if "pattern_type" in daily:
            pattern_type = daily["pattern_type"]
            if pattern_type == "weekday_heavy":
                insights.append("📅 Higher usage on weekdays - business/commercial area pattern")
            elif pattern_type == "weekend_heavy":
                insights.append("📅 Higher usage on weekends - residential/entertainment area pattern")
            else:
                insights.append("📅 Balanced usage throughout the week")
        
        # Congestion pattern insights
        congestion = pattern_analysis.get("congestion_patterns", {})
        if congestion.get("total_congestion_events", 0) > 0:
            congestion_rate = congestion.get("congestion_rate", 0)
            if congestion_rate > 10:
                insights.append(f"🚨 High congestion rate ({congestion_rate:.1f}%) indicates capacity issues")
            else:
                insights.append(f"⚠️ Moderate congestion rate ({congestion_rate:.1f}%)")
        
        # Capacity trend insights
        capacity = pattern_analysis.get("capacity_trends", {})
        if "trend" in capacity:
            trend = capacity["trend"]
            if trend == "increasing":
                growth_rate = capacity.get("growth_rate_percent")
                if growth_rate:
                    insights.append(f"📈 Capacity utilization increasing at {growth_rate}% rate")
                else:
                    insights.append("📈 Capacity utilization is increasing over time")
            elif trend == "decreasing":
                insights.append("📉 Capacity utilization is decreasing - possible reduced demand")
            else:
                insights.append("📊 Capacity utilization is stable")
        
        return insights

# Initialize tower load monitor
tower_monitor = TowerLoadMonitor()

# MCP Server Setup
logging.info("Creating Tower Load MCP Server instance...")
app = Server("tower-load-mcp-server")

# Wrap monitoring functions as ADK FunctionTools
ADK_TOWER_TOOLS = {
    "monitor_tower_load": FunctionTool(func=tower_monitor.monitor_tower_load),
    "detect_congestion": FunctionTool(func=tower_monitor.detect_congestion),
    "get_load_history": FunctionTool(func=tower_monitor.get_load_history),
    "classify_congestion_severity": FunctionTool(func=tower_monitor.classify_congestion_severity),
    "configure_congestion_thresholds": FunctionTool(func=tower_monitor.configure_congestion_thresholds),
    "detect_congestion_advanced": FunctionTool(func=tower_monitor.detect_congestion_advanced),
    "start_real_time_monitoring": FunctionTool(func=tower_monitor.start_real_time_monitoring),
    "stop_real_time_monitoring": FunctionTool(func=tower_monitor.stop_real_time_monitoring),
    "get_monitoring_status": FunctionTool(func=tower_monitor.get_monitoring_status),
    "get_load_history_advanced": FunctionTool(func=tower_monitor.get_load_history_advanced),
    "generate_performance_report": FunctionTool(func=tower_monitor.generate_performance_report),
    "analyze_network_patterns": FunctionTool(func=tower_monitor.analyze_network_patterns),
}

@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list tools this server exposes."""
    logging.info("Tower Load MCP Server: Received list_tools request.")
    mcp_tools_list = []
    for tool_name, adk_tool_instance in ADK_TOWER_TOOLS.items():
        if not adk_tool_instance.name:
            adk_tool_instance.name = tool_name

        mcp_tool_schema = adk_to_mcp_tool_type(adk_tool_instance)
        logging.info(f"Tower Load MCP Server: Advertising tool: {mcp_tool_schema.name}")
        mcp_tools_list.append(mcp_tool_schema)
    return mcp_tools_list

@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """MCP handler to execute a tool call requested by an MCP client."""
    logging.info(f"Tower Load MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    if name in ADK_TOWER_TOOLS:
        adk_tool_instance = ADK_TOWER_TOOLS[name]
        try:
            adk_tool_response = await adk_tool_instance.run_async(
                args=arguments,
                tool_context=None,
            )
            logging.info(f"Tower Load MCP Server: Tool '{name}' executed successfully")
            response_text = json.dumps(adk_tool_response, indent=2)
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
    else:
        logging.warning(f"Tower Load MCP Server: Tool '{name}' not found")
        error_payload = {
            "success": False,
            "message": f"Tool '{name}' not implemented by this server.",
            "available_tools": list(ADK_TOWER_TOOLS.keys())
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