"""
Spectrum Allocation MCP Server
Manages dynamic spectrum allocation and bandwidth optimization for 5G towers
Implements spectrum usage analysis, bandwidth reallocation, and allocation effectiveness monitoring
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
from network_models import (
    SpectrumAllocation, OptimizationAction, OptimizationActionType, ActionStatus,
    Tower, TowerMetrics, SeverityLevel
)
from network_db_utils import NetworkDatabaseManager
from dynamic_reallocation import DynamicReallocationEngine

load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Spectrum Allocation MCP] - %(message)s',
    handlers=[logging.StreamHandler()]
)

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")

class SpectrumManager:
    """Core spectrum allocation and management functionality"""
    
    def __init__(self):
        self.db_manager = NetworkDatabaseManager()
        self.reallocation_engine = DynamicReallocationEngine(self.db_manager)
        self.frequency_bands = {
            "low_band": {"range": "600-900 MHz", "bandwidth": 20.0, "coverage": "wide"},
            "mid_band": {"range": "2.5-3.7 GHz", "bandwidth": 100.0, "coverage": "medium"},
            "high_band": {"range": "24-39 GHz", "bandwidth": 400.0, "coverage": "narrow"}
        }
        self.default_allocation_duration = 3600  # 1 hour in seconds
        
    def analyze_spectrum_usage(self, tower_id: str) -> Dict[str, Any]:
        """Analyze current spectrum utilization for a specific tower"""
        try:
            # Get tower information
            tower = self.db_manager.get_tower_by_id(tower_id)
            if not tower:
                return {
                    "success": False,
                    "message": f"Tower {tower_id} not found",
                    "tower_id": tower_id
                }
            
            # Get current spectrum allocations
            allocations = self._get_current_allocations(tower_id)
            
            # Get latest tower metrics for utilization analysis
            metrics = self.db_manager.get_latest_tower_metrics(tower_id)
            if not metrics:
                return {
                    "success": False,
                    "message": f"No current metrics found for tower {tower_id}",
                    "tower_id": tower_id
                }
            
            # Calculate spectrum utilization statistics
            total_allocated = sum(alloc.allocated_bandwidth for alloc in allocations)
            total_available = sum(band["bandwidth"] for band in self.frequency_bands.values())
            utilization_percentage = (total_allocated / total_available) * 100 if total_available > 0 else 0
            
            # Analyze efficiency based on current load
            efficiency_score = self._calculate_spectrum_efficiency(allocations, metrics)
            
            # Generate recommendations
            recommendations = self._generate_spectrum_recommendations(tower_id, allocations, metrics)
            
            result = {
                "success": True,
                "tower_id": tower_id,
                "tower_name": tower.name,
                "analysis_timestamp": datetime.now().isoformat(),
                "spectrum_summary": {
                    "total_allocated_bandwidth": total_allocated,
                    "total_available_bandwidth": total_available,
                    "utilization_percentage": round(utilization_percentage, 2),
                    "efficiency_score": efficiency_score,
                    "active_allocations": len(allocations)
                },
                "frequency_band_analysis": self._analyze_frequency_bands(allocations),
                "current_allocations": [alloc.to_dict() for alloc in allocations],
                "tower_load_metrics": {
                    "cpu_utilization": metrics.cpu_utilization,
                    "memory_usage": metrics.memory_usage,
                    "bandwidth_usage": metrics.bandwidth_usage,
                    "active_connections": metrics.active_connections
                },
                "recommendations": recommendations
            }
            
            logging.info(f"Analyzed spectrum usage for tower {tower_id}: "
                        f"{utilization_percentage:.1f}% utilized, efficiency={efficiency_score:.1f}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error analyzing spectrum usage for tower {tower_id}: {e}")
            return {
                "success": False,
                "message": f"Error analyzing spectrum usage: {str(e)}",
                "tower_id": tower_id
            }
    
    def calculate_optimal_allocation(self, tower_ids: List[str], target_improvement: float = 20.0) -> Dict[str, Any]:
        """Calculate optimal bandwidth distribution across multiple towers"""
        try:
            if not tower_ids:
                return {
                    "success": False,
                    "message": "No tower IDs provided for allocation calculation"
                }
            
            # Validate towers exist
            towers = []
            for tower_id in tower_ids:
                tower = self.db_manager.get_tower_by_id(tower_id)
                if not tower:
                    return {
                        "success": False,
                        "message": f"Tower {tower_id} not found",
                        "invalid_tower_id": tower_id
                    }
                towers.append(tower)
            
            # Get current metrics and allocations for all towers
            tower_data = []
            for tower in towers:
                metrics = self.db_manager.get_latest_tower_metrics(tower.id)
                allocations = self._get_current_allocations(tower.id)
                
                if metrics:
                    tower_data.append({
                        "tower": tower,
                        "metrics": metrics,
                        "allocations": allocations,
                        "current_load": max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage),
                        "congestion_severity": metrics.congestion_severity
                    })
            
            if not tower_data:
                return {
                    "success": False,
                    "message": "No valid tower data found for optimization"
                }
            
            # Calculate optimal allocation plan
            allocation_plan = self._calculate_allocation_plan(tower_data, target_improvement)
            
            # Estimate execution time and effectiveness
            execution_estimate = self._estimate_execution_time(allocation_plan)
            effectiveness_estimate = self._estimate_effectiveness(allocation_plan, tower_data)
            
            result = {
                "success": True,
                "calculation_timestamp": datetime.now().isoformat(),
                "target_improvement": target_improvement,
                "towers_analyzed": len(tower_data),
                "allocation_plan": {
                    "plan_id": f"spectrum_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "total_actions": len(allocation_plan["actions"]),
                    "affected_towers": tower_ids,
                    "actions": allocation_plan["actions"],
                    "execution_time_estimate_minutes": execution_estimate,
                    "effectiveness_estimate": effectiveness_estimate
                },
                "current_state_analysis": [
                    {
                        "tower_id": data["tower"].id,
                        "tower_name": data["tower"].name,
                        "current_load": data["current_load"],
                        "congestion_severity": data["congestion_severity"].value,
                        "allocated_bandwidth": sum(a.allocated_bandwidth for a in data["allocations"]),
                        "utilization_efficiency": self._calculate_spectrum_efficiency(data["allocations"], data["metrics"])
                    }
                    for data in tower_data
                ],
                "optimization_summary": allocation_plan["summary"]
            }
            
            logging.info(f"Calculated optimal allocation for {len(tower_ids)} towers: "
                        f"{len(allocation_plan['actions'])} actions, "
                        f"{effectiveness_estimate:.1f}% expected improvement")
            
            return result
            
        except Exception as e:
            logging.error(f"Error calculating optimal allocation: {e}")
            return {
                "success": False,
                "message": f"Error calculating optimal allocation: {str(e)}",
                "tower_ids": tower_ids
            } 
   
    def execute_reallocation(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute spectrum reallocation based on provided plan"""
        try:
            plan_id = plan_data.get("plan_id", f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            actions = plan_data.get("actions", [])
            
            if not actions:
                return {
                    "success": False,
                    "message": "No actions provided in reallocation plan",
                    "plan_id": plan_id
                }
            
            execution_results = []
            successful_actions = 0
            failed_actions = 0
            
            for action_data in actions:
                try:
                    # Execute individual reallocation action
                    action_result = self._execute_single_action(action_data)
                    execution_results.append(action_result)
                    
                    if action_result["success"]:
                        successful_actions += 1
                        
                        # Log the optimization action to database
                        optimization_action = OptimizationAction(
                            action_type=OptimizationActionType.SPECTRUM_REALLOCATION,
                            tower_ids=action_data.get("tower_ids", []),
                            parameters=action_data,
                            executed_at=datetime.now(),
                            status=ActionStatus.COMPLETED,
                            effectiveness_score=action_result.get("effectiveness_score")
                        )
                        
                        self.db_manager.insert_optimization_action(optimization_action)
                        
                    else:
                        failed_actions += 1
                        
                except Exception as action_error:
                    logging.error(f"Error executing action {action_data}: {action_error}")
                    execution_results.append({
                        "success": False,
                        "action": action_data,
                        "error": str(action_error)
                    })
                    failed_actions += 1
            
            # Calculate overall execution effectiveness
            overall_effectiveness = (successful_actions / len(actions)) * 100 if actions else 0
            
            result = {
                "success": successful_actions > 0,
                "plan_id": plan_id,
                "execution_timestamp": datetime.now().isoformat(),
                "execution_summary": {
                    "total_actions": len(actions),
                    "successful_actions": successful_actions,
                    "failed_actions": failed_actions,
                    "overall_effectiveness": round(overall_effectiveness, 2)
                },
                "action_results": execution_results,
                "post_execution_analysis": self._analyze_post_execution_state(actions)
            }
            
            # Log network event
            self.db_manager.log_network_event(
                event_type="spectrum_reallocation",
                tower_id=None,
                severity="INFO" if successful_actions > 0 else "WARNING",
                description=f"Executed spectrum reallocation plan {plan_id}: "
                           f"{successful_actions}/{len(actions)} actions successful",
                metadata={"plan_id": plan_id, "effectiveness": overall_effectiveness}
            )
            
            logging.info(f"Executed reallocation plan {plan_id}: "
                        f"{successful_actions}/{len(actions)} actions successful")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing reallocation plan: {e}")
            return {
                "success": False,
                "message": f"Error executing reallocation plan: {str(e)}",
                "plan_id": plan_data.get("plan_id", "unknown")
            }
    
    def monitor_allocation_effectiveness(self, hours_back: int = 24) -> Dict[str, Any]:
        """Monitor effectiveness of recent spectrum allocations"""
        try:
            # Get recent optimization actions
            recent_actions = self.db_manager.get_recent_optimization_actions(hours_back)
            spectrum_actions = [a for a in recent_actions if a.action_type == OptimizationActionType.SPECTRUM_REALLOCATION]
            
            if not spectrum_actions:
                return {
                    "success": True,
                    "message": f"No spectrum reallocation actions found in the last {hours_back} hours",
                    "monitoring_period_hours": hours_back,
                    "total_actions": 0
                }
            
            # Analyze effectiveness of each action
            effectiveness_analysis = []
            total_effectiveness = 0
            
            for action in spectrum_actions:
                analysis = self._analyze_action_effectiveness(action)
                effectiveness_analysis.append(analysis)
                if analysis["effectiveness_score"] is not None:
                    total_effectiveness += analysis["effectiveness_score"]
            
            # Calculate overall statistics
            valid_scores = [a["effectiveness_score"] for a in effectiveness_analysis if a["effectiveness_score"] is not None]
            average_effectiveness = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            
            # Identify trends and patterns
            trends = self._identify_effectiveness_trends(effectiveness_analysis)
            
            # Generate improvement recommendations
            recommendations = self._generate_effectiveness_recommendations(effectiveness_analysis, trends)
            
            result = {
                "success": True,
                "monitoring_timestamp": datetime.now().isoformat(),
                "monitoring_period_hours": hours_back,
                "effectiveness_summary": {
                    "total_actions_analyzed": len(spectrum_actions),
                    "actions_with_valid_scores": len(valid_scores),
                    "average_effectiveness": round(average_effectiveness, 2),
                    "best_effectiveness": max(valid_scores) if valid_scores else 0,
                    "worst_effectiveness": min(valid_scores) if valid_scores else 0
                },
                "action_analysis": effectiveness_analysis,
                "trends": trends,
                "recommendations": recommendations,
                "performance_metrics": self._calculate_performance_metrics(spectrum_actions)
            }
            
            logging.info(f"Monitored allocation effectiveness: {len(spectrum_actions)} actions, "
                        f"average effectiveness {average_effectiveness:.1f}%")
            
            return result
            
        except Exception as e:
            logging.error(f"Error monitoring allocation effectiveness: {e}")
            return {
                "success": False,
                "message": f"Error monitoring allocation effectiveness: {str(e)}",
                "monitoring_period_hours": hours_back
            }
    
    def execute_dynamic_reallocation(self, tower_ids: List[str], algorithm: str = 'load_balancing') -> Dict[str, Any]:
        """Execute dynamic bandwidth reallocation using advanced algorithms"""
        try:
            if not tower_ids:
                return {
                    "success": False,
                    "message": "No tower IDs provided for dynamic reallocation"
                }
            
            # Prepare tower data for reallocation engine
            tower_data = []
            for tower_id in tower_ids:
                tower = self.db_manager.get_tower_by_id(tower_id)
                if not tower:
                    continue
                    
                metrics = self.db_manager.get_latest_tower_metrics(tower_id)
                allocations = self._get_current_allocations(tower_id)
                
                if metrics:
                    tower_data.append({
                        "tower": tower,
                        "metrics": metrics,
                        "allocations": allocations,
                        "current_load": max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage),
                        "congestion_severity": metrics.congestion_severity
                    })
            
            if not tower_data:
                return {
                    "success": False,
                    "message": "No valid tower data found for dynamic reallocation"
                }
            
            # Calculate optimal distribution using dynamic reallocation engine
            distribution_result = self.reallocation_engine.calculate_optimal_distribution(tower_data, algorithm)
            
            if not distribution_result["success"]:
                return distribution_result
            
            # Execute the reallocation plan
            execution_result = self.reallocation_engine.execute_reallocation_plan(
                distribution_result["distribution_plan"]
            )
            
            # Combine results
            result = {
                "success": execution_result["success"],
                "algorithm_used": algorithm,
                "execution_timestamp": datetime.now().isoformat(),
                "towers_processed": len(tower_data),
                "distribution_calculation": distribution_result,
                "execution_results": execution_result,
                "post_execution_analysis": self._analyze_reallocation_impact(tower_ids)
            }
            
            logging.info(f"Executed dynamic reallocation using {algorithm}: "
                        f"{execution_result['execution_summary']['successful_reallocations']} successful")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing dynamic reallocation: {e}")
            return {
                "success": False,
                "message": f"Error executing dynamic reallocation: {str(e)}",
                "algorithm": algorithm,
                "tower_ids": tower_ids
            }
    
    def monitor_reallocation_effectiveness_advanced(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Advanced monitoring of reallocation effectiveness with detailed analytics"""
        try:
            # Use the dynamic reallocation engine's monitoring capabilities
            effectiveness_result = self.reallocation_engine.monitor_reallocation_effectiveness(time_window_hours)
            
            if not effectiveness_result["success"]:
                return effectiveness_result
            
            # Add spectrum-specific analysis
            spectrum_analysis = self._analyze_spectrum_reallocation_patterns(time_window_hours)
            
            # Combine results
            result = effectiveness_result.copy()
            result["spectrum_specific_analysis"] = spectrum_analysis
            result["monitoring_timestamp"] = datetime.now().isoformat()
            
            logging.info(f"Advanced reallocation effectiveness monitoring completed: "
                        f"{effectiveness_result['effectiveness_summary']['total_actions_analyzed']} actions analyzed")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in advanced reallocation effectiveness monitoring: {e}")
            return {
                "success": False,
                "message": f"Error in advanced monitoring: {str(e)}",
                "monitoring_period_hours": time_window_hours
            }
    
    def optimize_spectrum_efficiency(self, tower_ids: List[str]) -> Dict[str, Any]:
        """Optimize spectrum efficiency across towers using advanced algorithms"""
        try:
            if not tower_ids:
                return {
                    "success": False,
                    "message": "No tower IDs provided for efficiency optimization"
                }
            
            # Prepare tower data
            tower_data = []
            for tower_id in tower_ids:
                tower = self.db_manager.get_tower_by_id(tower_id)
                if not tower:
                    continue
                    
                metrics = self.db_manager.get_latest_tower_metrics(tower_id)
                allocations = self._get_current_allocations(tower_id)
                
                if metrics:
                    tower_data.append({
                        "tower": tower,
                        "metrics": metrics,
                        "allocations": allocations,
                        "current_load": max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage),
                        "congestion_severity": metrics.congestion_severity
                    })
            
            if not tower_data:
                return {
                    "success": False,
                    "message": "No valid tower data found for efficiency optimization"
                }
            
            # Use efficiency optimization algorithm
            optimization_result = self.reallocation_engine.calculate_optimal_distribution(
                tower_data, 'efficiency_optimization'
            )
            
            if not optimization_result["success"]:
                return optimization_result
            
            # Execute optimization if there are actions to perform
            distribution_plan = optimization_result["distribution_plan"]
            if distribution_plan.get("reallocations"):
                execution_result = self.reallocation_engine.execute_reallocation_plan(distribution_plan)
                
                result = {
                    "success": execution_result["success"],
                    "optimization_timestamp": datetime.now().isoformat(),
                    "towers_optimized": len(tower_data),
                    "optimization_calculation": optimization_result,
                    "execution_results": execution_result,
                    "efficiency_improvements": self._calculate_efficiency_improvements(tower_ids)
                }
            else:
                result = {
                    "success": True,
                    "optimization_timestamp": datetime.now().isoformat(),
                    "towers_analyzed": len(tower_data),
                    "message": "No efficiency optimizations needed - spectrum allocation is already optimal",
                    "optimization_calculation": optimization_result
                }
            
            logging.info(f"Spectrum efficiency optimization completed for {len(tower_data)} towers")
            
            return result
            
        except Exception as e:
            logging.error(f"Error optimizing spectrum efficiency: {e}")
            return {
                "success": False,
                "message": f"Error optimizing spectrum efficiency: {str(e)}",
                "tower_ids": tower_ids
            }
    
    def execute_cross_tower_load_balancing(self, congested_tower_id: str, neighbor_radius_km: float = 5.0) -> Dict[str, Any]:
        """Execute cross-tower load balancing for congestion relief (Requirement 3.4)"""
        try:
            # Get congested tower information
            congested_tower = self.db_manager.get_tower_by_id(congested_tower_id)
            if not congested_tower:
                return {
                    "success": False,
                    "message": f"Congested tower {congested_tower_id} not found"
                }
            
            # Get current metrics for congested tower
            congested_metrics = self.db_manager.get_latest_tower_metrics(congested_tower_id)
            if not congested_metrics or not congested_metrics.is_congested:
                return {
                    "success": False,
                    "message": f"Tower {congested_tower_id} is not currently congested"
                }
            
            # Find neighboring towers within radius
            neighboring_towers = self._find_neighboring_towers(congested_tower, neighbor_radius_km)
            
            if not neighboring_towers:
                return {
                    "success": False,
                    "message": f"No neighboring towers found within {neighbor_radius_km}km radius"
                }
            
            # Analyze load balancing opportunities
            load_balancing_plan = self._calculate_load_balancing_plan(
                congested_tower, congested_metrics, neighboring_towers
            )
            
            if not load_balancing_plan["feasible"]:
                return {
                    "success": False,
                    "message": "Load balancing not feasible with current neighboring tower capacity",
                    "analysis": load_balancing_plan
                }
            
            # Execute load balancing with SLA compliance monitoring
            execution_result = self._execute_load_balancing_with_sla_monitoring(load_balancing_plan)
            
            # Log the load balancing action
            optimization_action = OptimizationAction(
                action_type=OptimizationActionType.LOAD_BALANCING,
                tower_ids=[congested_tower_id] + [t["tower"].id for t in neighboring_towers],
                parameters=load_balancing_plan,
                executed_at=datetime.now(),
                status=ActionStatus.COMPLETED if execution_result["success"] else ActionStatus.FAILED,
                effectiveness_score=execution_result.get("effectiveness_score")
            )
            
            self.db_manager.insert_optimization_action(optimization_action)
            
            result = {
                "success": execution_result["success"],
                "load_balancing_timestamp": datetime.now().isoformat(),
                "congested_tower": congested_tower_id,
                "neighboring_towers_used": len([t for t in neighboring_towers if t["selected_for_balancing"]]),
                "load_balancing_plan": load_balancing_plan,
                "execution_results": execution_result,
                "sla_compliance": execution_result.get("sla_compliance", {}),
                "post_balancing_analysis": self._analyze_post_balancing_state(congested_tower_id, neighboring_towers)
            }
            
            logging.info(f"Cross-tower load balancing executed for {congested_tower_id}: "
                        f"{'successful' if execution_result['success'] else 'failed'}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing cross-tower load balancing: {e}")
            return {
                "success": False,
                "message": f"Error executing cross-tower load balancing: {str(e)}",
                "congested_tower_id": congested_tower_id
            }
    
    def execute_traffic_redirection(self, source_tower_id: str, target_tower_ids: List[str], 
                                  traffic_percentage: float = 30.0) -> Dict[str, Any]:
        """Execute traffic redirection for congestion relief with SLA monitoring"""
        try:
            if not target_tower_ids:
                return {
                    "success": False,
                    "message": "No target towers specified for traffic redirection"
                }
            
            if not (10.0 <= traffic_percentage <= 50.0):
                return {
                    "success": False,
                    "message": "Traffic percentage must be between 10% and 50% for safe redirection"
                }
            
            # Validate source tower
            source_tower = self.db_manager.get_tower_by_id(source_tower_id)
            if not source_tower:
                return {
                    "success": False,
                    "message": f"Source tower {source_tower_id} not found"
                }
            
            # Validate target towers and check capacity
            target_towers = []
            for target_id in target_tower_ids:
                target_tower = self.db_manager.get_tower_by_id(target_id)
                if target_tower:
                    target_metrics = self.db_manager.get_latest_tower_metrics(target_id)
                    if target_metrics:
                        target_towers.append({
                            "tower": target_tower,
                            "metrics": target_metrics,
                            "available_capacity": 100 - max(target_metrics.cpu_utilization, 
                                                          target_metrics.memory_usage, 
                                                          target_metrics.bandwidth_usage)
                        })
            
            if not target_towers:
                return {
                    "success": False,
                    "message": "No valid target towers found for traffic redirection"
                }
            
            # Calculate traffic redirection plan
            redirection_plan = self._calculate_traffic_redirection_plan(
                source_tower_id, target_towers, traffic_percentage
            )
            
            if not redirection_plan["feasible"]:
                return {
                    "success": False,
                    "message": "Traffic redirection not feasible with current target tower capacity",
                    "analysis": redirection_plan
                }
            
            # Execute traffic redirection with SLA compliance monitoring
            execution_result = self._execute_traffic_redirection_with_sla_monitoring(redirection_plan)
            
            # Log the traffic redirection action
            optimization_action = OptimizationAction(
                action_type=OptimizationActionType.TRAFFIC_REDIRECT,
                tower_ids=[source_tower_id] + target_tower_ids,
                parameters=redirection_plan,
                executed_at=datetime.now(),
                status=ActionStatus.COMPLETED if execution_result["success"] else ActionStatus.FAILED,
                effectiveness_score=execution_result.get("effectiveness_score")
            )
            
            self.db_manager.insert_optimization_action(optimization_action)
            
            result = {
                "success": execution_result["success"],
                "redirection_timestamp": datetime.now().isoformat(),
                "source_tower": source_tower_id,
                "target_towers": target_tower_ids,
                "traffic_percentage_redirected": traffic_percentage,
                "redirection_plan": redirection_plan,
                "execution_results": execution_result,
                "sla_compliance": execution_result.get("sla_compliance", {}),
                "post_redirection_analysis": self._analyze_post_redirection_state(source_tower_id, target_tower_ids)
            }
            
            logging.info(f"Traffic redirection executed from {source_tower_id} to {len(target_tower_ids)} towers: "
                        f"{'successful' if execution_result['success'] else 'failed'}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing traffic redirection: {e}")
            return {
                "success": False,
                "message": f"Error executing traffic redirection: {str(e)}",
                "source_tower_id": source_tower_id,
                "target_tower_ids": target_tower_ids
            }
    
    async def monitor_sla_compliance_during_optimization(self, optimization_id: str, 
                                                       monitoring_duration_minutes: int = 10) -> Dict[str, Any]:
        """Monitor SLA compliance during optimization activities (Requirements 6.1, 6.2)"""
        try:
            # Get optimization action details
            optimization_action = self.db_manager.get_optimization_action_by_id(optimization_id)
            if not optimization_action:
                return {
                    "success": False,
                    "message": f"Optimization action {optimization_id} not found"
                }
            
            affected_towers = optimization_action.tower_ids
            monitoring_start = datetime.now()
            monitoring_end = monitoring_start + timedelta(minutes=monitoring_duration_minutes)
            
            sla_metrics = {
                "service_availability": [],
                "data_speed_compliance": [],
                "call_drop_rates": [],
                "handover_success_rates": []
            }
            
            # Monitor SLA metrics during the specified duration
            while datetime.now() < monitoring_end:
                for tower_id in affected_towers:
                    current_metrics = self.db_manager.get_latest_tower_metrics(tower_id)
                    if current_metrics:
                        # Calculate service availability (99.9% requirement)
                        service_availability = self._calculate_service_availability(current_metrics)
                        sla_metrics["service_availability"].append({
                            "tower_id": tower_id,
                            "timestamp": datetime.now().isoformat(),
                            "availability_percentage": service_availability
                        })
                        
                        # Monitor data speed compliance (30-second tolerance)
                        speed_compliance = self._check_data_speed_compliance(current_metrics)
                        sla_metrics["data_speed_compliance"].append({
                            "tower_id": tower_id,
                            "timestamp": datetime.now().isoformat(),
                            "speed_compliance": speed_compliance
                        })
                        
                        # Monitor call drop rates (0.1% threshold)
                        call_drop_rate = self._calculate_call_drop_rate(current_metrics)
                        sla_metrics["call_drop_rates"].append({
                            "tower_id": tower_id,
                            "timestamp": datetime.now().isoformat(),
                            "drop_rate_percentage": call_drop_rate
                        })
                
                # Wait before next measurement
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Analyze SLA compliance results
            compliance_analysis = self._analyze_sla_compliance(sla_metrics)
            
            result = {
                "success": True,
                "monitoring_timestamp": datetime.now().isoformat(),
                "optimization_id": optimization_id,
                "monitoring_duration_minutes": monitoring_duration_minutes,
                "affected_towers": affected_towers,
                "sla_metrics": sla_metrics,
                "compliance_analysis": compliance_analysis,
                "overall_compliance": compliance_analysis["overall_compliant"],
                "violations": compliance_analysis.get("violations", []),
                "recommendations": compliance_analysis.get("recommendations", [])
            }
            
            logging.info(f"SLA compliance monitoring completed for optimization {optimization_id}: "
                        f"{'compliant' if compliance_analysis['overall_compliant'] else 'violations detected'}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error monitoring SLA compliance: {e}")
            return {
                "success": False,
                "message": f"Error monitoring SLA compliance: {str(e)}",
                "optimization_id": optimization_id
            }
    
    def _analyze_reallocation_impact(self, tower_ids: List[str]) -> Dict[str, Any]:
        """Analyze the impact of recent reallocations on specified towers"""
        impact_analysis = {
            "towers_analyzed": len(tower_ids),
            "tower_impacts": [],
            "overall_impact": {}
        }
        
        total_load_improvement = 0
        total_efficiency_improvement = 0
        
        for tower_id in tower_ids:
            try:
                # Get current state
                current_metrics = self.db_manager.get_latest_tower_metrics(tower_id)
                current_allocations = self._get_current_allocations(tower_id)
                
                if current_metrics:
                    current_load = max(current_metrics.cpu_utilization, 
                                     current_metrics.memory_usage, 
                                     current_metrics.bandwidth_usage)
                    
                    efficiency_score = self._calculate_spectrum_efficiency(current_allocations, current_metrics)
                    
                    tower_impact = {
                        "tower_id": tower_id,
                        "current_load": current_load,
                        "efficiency_score": efficiency_score,
                        "total_allocated_bandwidth": sum(a.allocated_bandwidth for a in current_allocations),
                        "allocation_count": len(current_allocations),
                        "congestion_status": current_metrics.congestion_severity.value
                    }
                    
                    impact_analysis["tower_impacts"].append(tower_impact)
                    total_load_improvement += max(0, 80 - current_load)  # Improvement from 80% baseline
                    total_efficiency_improvement += efficiency_score
                    
            except Exception as e:
                logging.error(f"Error analyzing impact for tower {tower_id}: {e}")
        
        # Calculate overall impact metrics
        if impact_analysis["tower_impacts"]:
            impact_analysis["overall_impact"] = {
                "average_load": sum(t["current_load"] for t in impact_analysis["tower_impacts"]) / len(impact_analysis["tower_impacts"]),
                "average_efficiency": sum(t["efficiency_score"] for t in impact_analysis["tower_impacts"]) / len(impact_analysis["tower_impacts"]),
                "total_bandwidth_allocated": sum(t["total_allocated_bandwidth"] for t in impact_analysis["tower_impacts"]),
                "congestion_distribution": self._calculate_congestion_distribution(impact_analysis["tower_impacts"])
            }
        
        return impact_analysis
    
    # Load Balancing Helper Methods
    
    def _find_neighboring_towers(self, center_tower: Tower, radius_km: float) -> List[Dict[str, Any]]:
        """Find neighboring towers within specified radius"""
        try:
            all_towers = self.db_manager.get_all_towers()
            neighboring_towers = []
            
            for tower in all_towers:
                if tower.id == center_tower.id:
                    continue
                
                # Calculate distance using Haversine formula (simplified)
                distance = self._calculate_distance(
                    center_tower.latitude, center_tower.longitude,
                    tower.latitude, tower.longitude
                )
                
                if distance <= radius_km:
                    metrics = self.db_manager.get_latest_tower_metrics(tower.id)
                    if metrics:
                        neighboring_towers.append({
                            "tower": tower,
                            "metrics": metrics,
                            "distance_km": distance,
                            "current_load": max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage),
                            "available_capacity": 100 - max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage),
                            "selected_for_balancing": False
                        })
            
            # Sort by available capacity (descending) and distance (ascending)
            neighboring_towers.sort(key=lambda x: (-x["available_capacity"], x["distance_km"]))
            
            return neighboring_towers
            
        except Exception as e:
            logging.error(f"Error finding neighboring towers: {e}")
            return []
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        import math
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def _calculate_load_balancing_plan(self, congested_tower: Tower, congested_metrics: TowerMetrics, 
                                     neighboring_towers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate load balancing plan for congested tower"""
        try:
            congestion_level = max(congested_metrics.cpu_utilization, 
                                 congested_metrics.memory_usage, 
                                 congested_metrics.bandwidth_usage)
            
            # Calculate load to redistribute (aim to bring below 75%)
            target_load = 75.0
            load_to_redistribute = max(0, congestion_level - target_load)
            
            if load_to_redistribute < 5.0:  # Not worth redistributing small amounts
                return {
                    "feasible": False,
                    "reason": "Congestion level too low for load balancing",
                    "congestion_level": congestion_level
                }
            
            # Select neighboring towers for load balancing
            selected_towers = []
            remaining_load = load_to_redistribute
            
            for neighbor in neighboring_towers:
                if remaining_load <= 0:
                    break
                
                # Check if neighbor can accept load (keep below 80% after transfer)
                max_acceptable_load = min(neighbor["available_capacity"] * 0.8, remaining_load)
                
                if max_acceptable_load >= 5.0:  # Minimum 5% load transfer
                    selected_towers.append({
                        "tower_id": neighbor["tower"].id,
                        "tower_name": neighbor["tower"].name,
                        "distance_km": neighbor["distance_km"],
                        "load_to_accept": max_acceptable_load,
                        "current_load": neighbor["current_load"],
                        "post_transfer_load": neighbor["current_load"] + max_acceptable_load
                    })
                    
                    neighbor["selected_for_balancing"] = True
                    remaining_load -= max_acceptable_load
            
            feasible = len(selected_towers) > 0 and remaining_load < load_to_redistribute * 0.5
            
            plan = {
                "feasible": feasible,
                "congested_tower_id": congested_tower.id,
                "initial_congestion_level": congestion_level,
                "target_load": target_load,
                "total_load_to_redistribute": load_to_redistribute,
                "load_redistributed": load_to_redistribute - remaining_load,
                "remaining_load": remaining_load,
                "selected_towers": selected_towers,
                "estimated_post_balancing_load": congestion_level - (load_to_redistribute - remaining_load),
                "sla_risk_assessment": self._assess_load_balancing_sla_risk(selected_towers)
            }
            
            return plan
            
        except Exception as e:
            logging.error(f"Error calculating load balancing plan: {e}")
            return {"feasible": False, "error": str(e)}
    
    def _calculate_traffic_redirection_plan(self, source_tower_id: str, target_towers: List[Dict[str, Any]], 
                                          traffic_percentage: float) -> Dict[str, Any]:
        """Calculate traffic redirection plan"""
        try:
            # Calculate total traffic to redirect
            source_metrics = self.db_manager.get_latest_tower_metrics(source_tower_id)
            if not source_metrics:
                return {"feasible": False, "reason": "Source tower metrics not available"}
            
            current_connections = source_metrics.active_connections
            connections_to_redirect = int(current_connections * (traffic_percentage / 100))
            
            # Distribute redirected traffic across target towers
            total_target_capacity = sum(t["available_capacity"] for t in target_towers)
            
            if total_target_capacity < traffic_percentage:  # Not enough capacity
                return {
                    "feasible": False,
                    "reason": "Insufficient capacity in target towers",
                    "required_capacity": traffic_percentage,
                    "available_capacity": total_target_capacity
                }
            
            # Calculate distribution
            redirection_distribution = []
            remaining_connections = connections_to_redirect
            
            for target in target_towers:
                if remaining_connections <= 0:
                    break
                
                # Distribute proportionally based on available capacity
                capacity_ratio = target["available_capacity"] / total_target_capacity
                connections_for_this_target = min(
                    int(connections_to_redirect * capacity_ratio),
                    remaining_connections,
                    int(target["available_capacity"] * target["tower"].max_capacity / 100)
                )
                
                if connections_for_this_target > 0:
                    redirection_distribution.append({
                        "target_tower_id": target["tower"].id,
                        "target_tower_name": target["tower"].name,
                        "connections_to_accept": connections_for_this_target,
                        "current_load": target["metrics"].bandwidth_usage,
                        "estimated_post_redirect_load": target["metrics"].bandwidth_usage + 
                                                      (connections_for_this_target / target["tower"].max_capacity * 100)
                    })
                    
                    remaining_connections -= connections_for_this_target
            
            plan = {
                "feasible": len(redirection_distribution) > 0,
                "source_tower_id": source_tower_id,
                "traffic_percentage": traffic_percentage,
                "total_connections_to_redirect": connections_to_redirect,
                "connections_redistributed": connections_to_redirect - remaining_connections,
                "redirection_distribution": redirection_distribution,
                "sla_risk_assessment": self._assess_traffic_redirection_sla_risk(redirection_distribution)
            }
            
            return plan
            
        except Exception as e:
            logging.error(f"Error calculating traffic redirection plan: {e}")
            return {"feasible": False, "error": str(e)}
    
    def _execute_load_balancing_with_sla_monitoring(self, load_balancing_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute load balancing with SLA compliance monitoring"""
        try:
            if not load_balancing_plan.get("feasible"):
                return {
                    "success": False,
                    "message": "Load balancing plan is not feasible"
                }
            
            execution_results = []
            sla_violations = []
            successful_transfers = 0
            
            # Execute load transfers to selected towers
            for selected_tower in load_balancing_plan["selected_towers"]:
                try:
                    # Simulate load transfer (in real implementation, this would interface with network equipment)
                    transfer_result = self._simulate_load_transfer(
                        load_balancing_plan["congested_tower_id"],
                        selected_tower["tower_id"],
                        selected_tower["load_to_accept"]
                    )
                    
                    # Monitor SLA compliance during transfer
                    sla_compliance = self._monitor_transfer_sla_compliance(
                        selected_tower["tower_id"],
                        selected_tower["load_to_accept"]
                    )
                    
                    if transfer_result["success"] and sla_compliance["compliant"]:
                        successful_transfers += 1
                        execution_results.append({
                            "tower_id": selected_tower["tower_id"],
                            "success": True,
                            "load_transferred": selected_tower["load_to_accept"],
                            "sla_compliance": sla_compliance
                        })
                    else:
                        execution_results.append({
                            "tower_id": selected_tower["tower_id"],
                            "success": False,
                            "error": transfer_result.get("error", "SLA violation"),
                            "sla_compliance": sla_compliance
                        })
                        
                        if not sla_compliance["compliant"]:
                            sla_violations.append(sla_compliance)
                
                except Exception as transfer_error:
                    execution_results.append({
                        "tower_id": selected_tower["tower_id"],
                        "success": False,
                        "error": str(transfer_error)
                    })
            
            # Calculate overall effectiveness
            effectiveness_score = (successful_transfers / len(load_balancing_plan["selected_towers"])) * 100
            
            result = {
                "success": successful_transfers > 0,
                "execution_timestamp": datetime.now().isoformat(),
                "successful_transfers": successful_transfers,
                "total_transfers_attempted": len(load_balancing_plan["selected_towers"]),
                "effectiveness_score": effectiveness_score,
                "execution_results": execution_results,
                "sla_compliance": {
                    "violations_detected": len(sla_violations),
                    "overall_compliant": len(sla_violations) == 0,
                    "violation_details": sla_violations
                }
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing load balancing: {e}")
            return {
                "success": False,
                "message": f"Error executing load balancing: {str(e)}"
            }
    
    def _execute_traffic_redirection_with_sla_monitoring(self, redirection_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute traffic redirection with SLA compliance monitoring"""
        try:
            if not redirection_plan.get("feasible"):
                return {
                    "success": False,
                    "message": "Traffic redirection plan is not feasible"
                }
            
            execution_results = []
            sla_violations = []
            successful_redirections = 0
            
            # Execute traffic redirections
            for redirection in redirection_plan["redirection_distribution"]:
                try:
                    # Simulate traffic redirection (in real implementation, this would interface with network equipment)
                    redirection_result = self._simulate_traffic_redirection(
                        redirection_plan["source_tower_id"],
                        redirection["target_tower_id"],
                        redirection["connections_to_accept"]
                    )
                    
                    # Monitor SLA compliance during redirection
                    sla_compliance = self._monitor_redirection_sla_compliance(
                        redirection["target_tower_id"],
                        redirection["connections_to_accept"]
                    )
                    
                    if redirection_result["success"] and sla_compliance["compliant"]:
                        successful_redirections += 1
                        execution_results.append({
                            "target_tower_id": redirection["target_tower_id"],
                            "success": True,
                            "connections_redirected": redirection["connections_to_accept"],
                            "sla_compliance": sla_compliance
                        })
                    else:
                        execution_results.append({
                            "target_tower_id": redirection["target_tower_id"],
                            "success": False,
                            "error": redirection_result.get("error", "SLA violation"),
                            "sla_compliance": sla_compliance
                        })
                        
                        if not sla_compliance["compliant"]:
                            sla_violations.append(sla_compliance)
                
                except Exception as redirection_error:
                    execution_results.append({
                        "target_tower_id": redirection["target_tower_id"],
                        "success": False,
                        "error": str(redirection_error)
                    })
            
            # Calculate overall effectiveness
            effectiveness_score = (successful_redirections / len(redirection_plan["redirection_distribution"])) * 100
            
            result = {
                "success": successful_redirections > 0,
                "execution_timestamp": datetime.now().isoformat(),
                "successful_redirections": successful_redirections,
                "total_redirections_attempted": len(redirection_plan["redirection_distribution"]),
                "effectiveness_score": effectiveness_score,
                "execution_results": execution_results,
                "sla_compliance": {
                    "violations_detected": len(sla_violations),
                    "overall_compliant": len(sla_violations) == 0,
                    "violation_details": sla_violations
                }
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing traffic redirection: {e}")
            return {
                "success": False,
                "message": f"Error executing traffic redirection: {str(e)}"
            }
    
    def _analyze_spectrum_reallocation_patterns(self, time_window_hours: int) -> Dict[str, Any]:
        """Analyze spectrum-specific reallocation patterns"""
        try:
            recent_actions = self.db_manager.get_recent_optimization_actions(time_window_hours)
            spectrum_actions = [a for a in recent_actions if a.action_type == OptimizationActionType.SPECTRUM_REALLOCATION]
            
            if not spectrum_actions:
                return {"message": "No spectrum reallocation actions found", "patterns": {}}
            
            patterns = {
                "frequency_band_usage": {},
                "bandwidth_distribution": {"small": 0, "medium": 0, "large": 0},
                "algorithm_effectiveness": {},
                "time_patterns": {}
            }
            
            for action in spectrum_actions:
                if isinstance(action.parameters, dict):
                    # Analyze frequency band patterns
                    if "target_frequency_band" in action.parameters:
                        band = action.parameters["target_frequency_band"]
                        patterns["frequency_band_usage"][band] = patterns["frequency_band_usage"].get(band, 0) + 1
                    
                    # Analyze bandwidth distribution
                    bandwidth = action.parameters.get("bandwidth_amount", 0)
                    if bandwidth < 50:
                        patterns["bandwidth_distribution"]["small"] += 1
                    elif bandwidth < 150:
                        patterns["bandwidth_distribution"]["medium"] += 1
                    else:
                        patterns["bandwidth_distribution"]["large"] += 1
                    
                    # Analyze algorithm effectiveness
                    algorithm = action.parameters.get("algorithm", "unknown")
                    if algorithm not in patterns["algorithm_effectiveness"]:
                        patterns["algorithm_effectiveness"][algorithm] = {"count": 0, "total_effectiveness": 0}
                    
                    patterns["algorithm_effectiveness"][algorithm]["count"] += 1
                    if action.effectiveness_score:
                        patterns["algorithm_effectiveness"][algorithm]["total_effectiveness"] += action.effectiveness_score
                
                # Analyze time patterns
                hour = action.executed_at.hour
                patterns["time_patterns"][hour] = patterns["time_patterns"].get(hour, 0) + 1
            
            # Calculate averages for algorithm effectiveness
            for algorithm, data in patterns["algorithm_effectiveness"].items():
                if data["count"] > 0:
                    data["average_effectiveness"] = data["total_effectiveness"] / data["count"]
            
            return {"patterns": patterns, "total_actions_analyzed": len(spectrum_actions)}
            
        except Exception as e:
            logging.error(f"Error analyzing spectrum reallocation patterns: {e}")
            return {"error": str(e)}
    
    def _calculate_efficiency_improvements(self, tower_ids: List[str]) -> Dict[str, Any]:
        """Calculate efficiency improvements after optimization"""
        improvements = {
            "tower_improvements": [],
            "overall_improvement": 0
        }
        
        total_improvement = 0
        
        for tower_id in tower_ids:
            try:
                current_metrics = self.db_manager.get_latest_tower_metrics(tower_id)
                current_allocations = self._get_current_allocations(tower_id)
                
                if current_metrics and current_allocations:
                    efficiency_score = self._calculate_spectrum_efficiency(current_allocations, current_metrics)
                    
                    # Estimate improvement (simplified - would compare with pre-optimization state in real implementation)
                    estimated_improvement = max(0, efficiency_score - 60)  # Improvement over 60% baseline
                    
                    tower_improvement = {
                        "tower_id": tower_id,
                        "current_efficiency": efficiency_score,
                        "estimated_improvement": estimated_improvement,
                        "optimization_potential": max(0, 90 - efficiency_score)  # Potential for further improvement
                    }
                    
                    improvements["tower_improvements"].append(tower_improvement)
                    total_improvement += estimated_improvement
                    
            except Exception as e:
                logging.error(f"Error calculating efficiency improvement for {tower_id}: {e}")
        
        improvements["overall_improvement"] = total_improvement / len(tower_ids) if tower_ids else 0
        
        return improvements
    
    def _calculate_congestion_distribution(self, tower_impacts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of congestion levels"""
        distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "NONE": 0}
        
        for impact in tower_impacts:
            congestion_status = impact.get("congestion_status", "NONE")
            distribution[congestion_status] = distribution.get(congestion_status, 0) + 1
        
        return distribution
    
    # Helper methods for spectrum management
    
    def _get_current_allocations(self, tower_id: str) -> List[SpectrumAllocation]:
        """Get current spectrum allocations for a tower"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM spectrum_allocations 
                    WHERE tower_id = ? AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY allocation_timestamp DESC
                """, (tower_id, datetime.now().isoformat()))
                
                rows = cursor.fetchall()
                allocations = []
                
                for row in rows:
                    allocation = SpectrumAllocation(
                        id=row['id'],
                        tower_id=row['tower_id'],
                        frequency_band=row['frequency_band'],
                        allocated_bandwidth=row['allocated_bandwidth'],
                        utilization_percentage=row['utilization_percentage'],
                        allocation_timestamp=datetime.fromisoformat(row['allocation_timestamp']),
                        expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
                    )
                    allocations.append(allocation)
                
                return allocations
                
        except Exception as e:
            logging.error(f"Error getting current allocations for {tower_id}: {e}")
            return []
    
    def _calculate_spectrum_efficiency(self, allocations: List[SpectrumAllocation], metrics: TowerMetrics) -> float:
        """Calculate spectrum utilization efficiency score"""
        if not allocations:
            return 0.0
        
        # Calculate weighted efficiency based on bandwidth usage and allocation
        total_allocated = sum(alloc.allocated_bandwidth for alloc in allocations)
        if total_allocated == 0:
            return 0.0
        
        # Efficiency is based on how well allocated spectrum matches actual usage
        actual_usage = metrics.bandwidth_usage
        allocation_efficiency = min(actual_usage / 80.0, 1.0)  # Normalize to 80% as optimal
        
        # Factor in signal quality and error rates
        signal_quality_factor = max(0, (metrics.signal_strength + 100) / 30)  # -100 to -70 dBm range
        error_penalty = max(0, 1 - (metrics.error_rate / 10))  # Penalty for high error rates
        
        efficiency_score = allocation_efficiency * signal_quality_factor * error_penalty * 100
        return min(100.0, max(0.0, efficiency_score))
    
    def _analyze_frequency_bands(self, allocations: List[SpectrumAllocation]) -> Dict[str, Any]:
        """Analyze utilization across different frequency bands"""
        band_analysis = {}
        
        for band_name, band_info in self.frequency_bands.items():
            band_allocations = [a for a in allocations if band_name in a.frequency_band.lower()]
            total_allocated = sum(a.allocated_bandwidth for a in band_allocations)
            avg_utilization = sum(a.utilization_percentage for a in band_allocations) / len(band_allocations) if band_allocations else 0
            
            band_analysis[band_name] = {
                "total_available": band_info["bandwidth"],
                "total_allocated": total_allocated,
                "utilization_percentage": (total_allocated / band_info["bandwidth"]) * 100 if band_info["bandwidth"] > 0 else 0,
                "average_usage": round(avg_utilization, 2),
                "active_allocations": len(band_allocations),
                "coverage_type": band_info["coverage"],
                "frequency_range": band_info["range"]
            }
        
        return band_analysis
    
    def _generate_spectrum_recommendations(self, tower_id: str, allocations: List[SpectrumAllocation], 
                                         metrics: TowerMetrics) -> List[str]:
        """Generate spectrum optimization recommendations"""
        recommendations = []
        
        # Check for over-allocation
        total_allocated = sum(alloc.allocated_bandwidth for alloc in allocations)
        total_available = sum(band["bandwidth"] for band in self.frequency_bands.values())
        
        if total_allocated > total_available * 0.9:
            recommendations.append("Consider load balancing to neighboring towers - spectrum utilization is very high")
        
        # Check for under-utilization
        if metrics.bandwidth_usage < 50 and total_allocated > 100:
            recommendations.append("Reduce spectrum allocation - current usage is low compared to allocated bandwidth")
        
        # Check for congestion
        if metrics.bandwidth_usage > 85:
            recommendations.append("Increase spectrum allocation - tower is experiencing high bandwidth utilization")
        
        # Check signal quality
        if metrics.signal_strength < -85:
            recommendations.append("Consider frequency band reallocation - signal strength is poor")
        
        # Check error rates
        if metrics.error_rate > 2:
            recommendations.append("Investigate spectrum interference - error rate is elevated")
        
        # Band-specific recommendations
        band_analysis = self._analyze_frequency_bands(allocations)
        for band_name, analysis in band_analysis.items():
            if analysis["utilization_percentage"] > 90:
                recommendations.append(f"High utilization in {band_name} band - consider rebalancing")
            elif analysis["utilization_percentage"] < 20 and analysis["active_allocations"] > 0:
                recommendations.append(f"Low utilization in {band_name} band - consider reallocating to other bands")
        
        return recommendations if recommendations else ["No specific recommendations - spectrum allocation appears optimal"] 
   
    def _calculate_allocation_plan(self, tower_data: List[Dict], target_improvement: float) -> Dict[str, Any]:
        """Calculate optimal allocation plan for towers"""
        actions = []
        summary = {
            "total_bandwidth_reallocation": 0,
            "towers_optimized": 0,
            "expected_improvement": 0
        }
        
        # Sort towers by congestion severity and load
        sorted_towers = sorted(tower_data, key=lambda x: (
            x["congestion_severity"].value == "HIGH",
            x["congestion_severity"].value == "MEDIUM",
            x["current_load"]
        ), reverse=True)
        
        for tower_info in sorted_towers:
            tower = tower_info["tower"]
            metrics = tower_info["metrics"]
            current_allocations = tower_info["allocations"]
            
            # Calculate needed bandwidth adjustment
            if metrics.bandwidth_usage > 80:  # Congested
                # Increase allocation
                needed_increase = self._calculate_bandwidth_increase(metrics, current_allocations)
                if needed_increase > 0:
                    action = {
                        "action_type": "increase_allocation",
                        "tower_ids": [tower.id],
                        "current_bandwidth": sum(a.allocated_bandwidth for a in current_allocations),
                        "bandwidth_increase": needed_increase,
                        "target_frequency_band": self._select_optimal_band(tower.id, needed_increase),
                        "priority": "HIGH" if metrics.bandwidth_usage > 90 else "MEDIUM",
                        "expected_improvement": min(needed_increase / 10, target_improvement)
                    }
                    actions.append(action)
                    summary["total_bandwidth_reallocation"] += needed_increase
                    summary["towers_optimized"] += 1
                    
            elif metrics.bandwidth_usage < 40:  # Under-utilized
                # Decrease allocation and redistribute
                possible_decrease = self._calculate_bandwidth_decrease(metrics, current_allocations)
                if possible_decrease > 0:
                    action = {
                        "action_type": "decrease_allocation",
                        "tower_ids": [tower.id],
                        "current_bandwidth": sum(a.allocated_bandwidth for a in current_allocations),
                        "bandwidth_decrease": possible_decrease,
                        "redistribute_to": self._find_redistribution_targets(tower_data, tower.id),
                        "priority": "LOW",
                        "expected_improvement": possible_decrease / 20
                    }
                    actions.append(action)
                    summary["total_bandwidth_reallocation"] += possible_decrease
                    summary["towers_optimized"] += 1
        
        # Calculate overall expected improvement
        summary["expected_improvement"] = min(
            sum(action.get("expected_improvement", 0) for action in actions),
            target_improvement
        )
        
        return {"actions": actions, "summary": summary}
    
    def _calculate_bandwidth_increase(self, metrics: TowerMetrics, allocations: List[SpectrumAllocation]) -> float:
        """Calculate needed bandwidth increase for congested tower"""
        current_total = sum(a.allocated_bandwidth for a in allocations)
        usage_percentage = metrics.bandwidth_usage
        
        if usage_percentage > 90:
            return current_total * 0.5  # 50% increase for critical congestion
        elif usage_percentage > 80:
            return current_total * 0.3  # 30% increase for moderate congestion
        
        return 0.0
    
    def _calculate_bandwidth_decrease(self, metrics: TowerMetrics, allocations: List[SpectrumAllocation]) -> float:
        """Calculate possible bandwidth decrease for under-utilized tower"""
        current_total = sum(a.allocated_bandwidth for a in allocations)
        usage_percentage = metrics.bandwidth_usage
        
        if usage_percentage < 30:
            return current_total * 0.4  # 40% decrease for very low usage
        elif usage_percentage < 50:
            return current_total * 0.2  # 20% decrease for low usage
        
        return 0.0
    
    def _select_optimal_band(self, tower_id: str, bandwidth_needed: float) -> str:
        """Select optimal frequency band for allocation"""
        # Simple heuristic: prefer mid-band for balanced coverage and capacity
        if bandwidth_needed > 200:
            return "high_band"  # High capacity needed
        elif bandwidth_needed > 50:
            return "mid_band"   # Balanced option
        else:
            return "low_band"   # Wide coverage
    
    def _find_redistribution_targets(self, tower_data: List[Dict], source_tower_id: str) -> List[str]:
        """Find towers that could benefit from redistributed bandwidth"""
        targets = []
        for tower_info in tower_data:
            if (tower_info["tower"].id != source_tower_id and 
                tower_info["current_load"] > 70):
                targets.append(tower_info["tower"].id)
        return targets[:2]  # Limit to 2 targets
    
    def _estimate_execution_time(self, allocation_plan: Dict[str, Any]) -> int:
        """Estimate execution time in minutes"""
        actions = allocation_plan.get("actions", [])
        base_time_per_action = 2  # 2 minutes per action
        return len(actions) * base_time_per_action
    
    def _estimate_effectiveness(self, allocation_plan: Dict[str, Any], tower_data: List[Dict]) -> float:
        """Estimate effectiveness of allocation plan"""
        actions = allocation_plan.get("actions", [])
        if not actions:
            return 0.0
        
        total_expected = sum(action.get("expected_improvement", 0) for action in actions)
        return min(total_expected, 100.0)
    
    def _execute_single_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single spectrum allocation action"""
        try:
            action_type = action_data.get("action_type")
            tower_ids = action_data.get("tower_ids", [])
            
            if not tower_ids:
                return {
                    "success": False,
                    "action": action_data,
                    "error": "No tower IDs specified"
                }
            
            # Simulate spectrum allocation execution
            # In a real implementation, this would interface with actual network equipment
            
            if action_type == "increase_allocation":
                result = self._execute_bandwidth_increase(action_data)
            elif action_type == "decrease_allocation":
                result = self._execute_bandwidth_decrease(action_data)
            else:
                return {
                    "success": False,
                    "action": action_data,
                    "error": f"Unknown action type: {action_type}"
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "action": action_data,
                "error": str(e)
            }
    
    def _execute_bandwidth_increase(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bandwidth increase action"""
        tower_id = action_data["tower_ids"][0]
        bandwidth_increase = action_data.get("bandwidth_increase", 0)
        frequency_band = action_data.get("target_frequency_band", "mid_band")
        
        # Create new spectrum allocation record
        allocation = SpectrumAllocation(
            tower_id=tower_id,
            frequency_band=frequency_band,
            allocated_bandwidth=bandwidth_increase,
            allocation_timestamp=datetime.now(),
            utilization_percentage=0.0,  # Will be updated as usage is monitored
            expires_at=datetime.now() + timedelta(seconds=self.default_allocation_duration)
        )
        
        # Insert into database
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO spectrum_allocations 
                (tower_id, frequency_band, allocated_bandwidth, utilization_percentage, 
                 allocation_timestamp, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                allocation.tower_id, allocation.frequency_band, allocation.allocated_bandwidth,
                allocation.utilization_percentage, allocation.allocation_timestamp.isoformat(),
                allocation.expires_at.isoformat() if allocation.expires_at else None
            ))
            conn.commit()
            allocation.id = cursor.lastrowid
        
        # Calculate effectiveness score
        effectiveness_score = min((bandwidth_increase / 100) * 80, 100)  # Heuristic scoring
        
        return {
            "success": True,
            "action": action_data,
            "allocation_id": allocation.id,
            "allocated_bandwidth": bandwidth_increase,
            "frequency_band": frequency_band,
            "effectiveness_score": effectiveness_score,
            "execution_timestamp": datetime.now().isoformat()
        }
    
    def _execute_bandwidth_decrease(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bandwidth decrease action"""
        tower_id = action_data["tower_ids"][0]
        bandwidth_decrease = action_data.get("bandwidth_decrease", 0)
        
        # Find and update existing allocations
        current_allocations = self._get_current_allocations(tower_id)
        
        if not current_allocations:
            return {
                "success": False,
                "action": action_data,
                "error": "No current allocations found to decrease"
            }
        
        # Decrease from largest allocation first
        remaining_decrease = bandwidth_decrease
        updated_allocations = []
        
        for allocation in sorted(current_allocations, key=lambda a: a.allocated_bandwidth, reverse=True):
            if remaining_decrease <= 0:
                break
                
            decrease_amount = min(remaining_decrease, allocation.allocated_bandwidth * 0.8)  # Max 80% decrease
            new_bandwidth = allocation.allocated_bandwidth - decrease_amount
            
            # Update allocation in database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE spectrum_allocations 
                    SET allocated_bandwidth = ?, allocation_timestamp = ?
                    WHERE id = ?
                """, (new_bandwidth, datetime.now().isoformat(), allocation.id))
                conn.commit()
            
            updated_allocations.append({
                "allocation_id": allocation.id,
                "old_bandwidth": allocation.allocated_bandwidth,
                "new_bandwidth": new_bandwidth,
                "decrease_amount": decrease_amount
            })
            
            remaining_decrease -= decrease_amount
        
        effectiveness_score = min(((bandwidth_decrease - remaining_decrease) / bandwidth_decrease) * 70, 100)
        
        return {
            "success": True,
            "action": action_data,
            "updated_allocations": updated_allocations,
            "total_decreased": bandwidth_decrease - remaining_decrease,
            "effectiveness_score": effectiveness_score,
            "execution_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_post_execution_state(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze network state after executing actions"""
        affected_towers = set()
        for action in actions:
            affected_towers.update(action.get("tower_ids", []))
        
        post_analysis = {
            "affected_towers": len(affected_towers),
            "tower_states": []
        }
        
        for tower_id in affected_towers:
            # Get updated metrics and allocations
            metrics = self.db_manager.get_latest_tower_metrics(tower_id)
            allocations = self._get_current_allocations(tower_id)
            
            if metrics:
                tower_state = {
                    "tower_id": tower_id,
                    "current_load": max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage),
                    "total_allocated_bandwidth": sum(a.allocated_bandwidth for a in allocations),
                    "spectrum_efficiency": self._calculate_spectrum_efficiency(allocations, metrics),
                    "congestion_status": metrics.congestion_severity.value
                }
                post_analysis["tower_states"].append(tower_state)
        
        return post_analysis
    
    def _analyze_action_effectiveness(self, action: OptimizationAction) -> Dict[str, Any]:
        """Analyze effectiveness of a specific optimization action"""
        try:
            # Get tower metrics before and after the action
            action_time = action.executed_at
            before_time = action_time - timedelta(minutes=30)
            after_time = action_time + timedelta(minutes=30)
            
            effectiveness_data = {
                "action_id": action.id,
                "action_type": action.action_type.value,
                "tower_ids": action.tower_ids,
                "executed_at": action.executed_at.isoformat(),
                "effectiveness_score": action.effectiveness_score,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Analyze impact on each affected tower
            tower_impacts = []
            for tower_id in action.tower_ids:
                # Get metrics around the action time
                before_metrics = self._get_metrics_near_time(tower_id, before_time)
                after_metrics = self._get_metrics_near_time(tower_id, after_time)
                
                if before_metrics and after_metrics:
                    impact = self._calculate_tower_impact(before_metrics, after_metrics)
                    tower_impacts.append({
                        "tower_id": tower_id,
                        "before_load": max(before_metrics.cpu_utilization, 
                                         before_metrics.memory_usage, 
                                         before_metrics.bandwidth_usage),
                        "after_load": max(after_metrics.cpu_utilization, 
                                        after_metrics.memory_usage, 
                                        after_metrics.bandwidth_usage),
                        "load_improvement": impact["load_improvement"],
                        "bandwidth_improvement": impact["bandwidth_improvement"],
                        "signal_improvement": impact["signal_improvement"]
                    })
            
            effectiveness_data["tower_impacts"] = tower_impacts
            
            # Calculate overall impact score
            if tower_impacts:
                avg_load_improvement = sum(t["load_improvement"] for t in tower_impacts) / len(tower_impacts)
                effectiveness_data["calculated_effectiveness"] = max(0, min(100, avg_load_improvement))
            else:
                effectiveness_data["calculated_effectiveness"] = None
            
            return effectiveness_data
            
        except Exception as e:
            logging.error(f"Error analyzing action effectiveness: {e}")
            return {
                "action_id": action.id,
                "effectiveness_score": None,
                "error": str(e)
            }
    
    def _get_metrics_near_time(self, tower_id: str, target_time: datetime) -> Optional[TowerMetrics]:
        """Get tower metrics closest to target time"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tower_metrics 
                    WHERE tower_id = ? 
                    ORDER BY ABS(julianday(timestamp) - julianday(?))
                    LIMIT 1
                """, (tower_id, target_time.isoformat()))
                
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
                
        except Exception as e:
            logging.error(f"Error getting metrics near time for {tower_id}: {e}")
            return None
    
    def _calculate_tower_impact(self, before: TowerMetrics, after: TowerMetrics) -> Dict[str, float]:
        """Calculate impact metrics between before and after states"""
        before_load = max(before.cpu_utilization, before.memory_usage, before.bandwidth_usage)
        after_load = max(after.cpu_utilization, after.memory_usage, after.bandwidth_usage)
        
        return {
            "load_improvement": before_load - after_load,
            "bandwidth_improvement": before.bandwidth_usage - after.bandwidth_usage,
            "signal_improvement": after.signal_strength - before.signal_strength,
            "error_improvement": before.error_rate - after.error_rate
        }
    
    def _identify_effectiveness_trends(self, effectiveness_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify trends in allocation effectiveness"""
        valid_scores = [a["effectiveness_score"] for a in effectiveness_analysis 
                       if a["effectiveness_score"] is not None]
        
        if len(valid_scores) < 2:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Simple trend analysis
        recent_scores = valid_scores[-5:]  # Last 5 actions
        older_scores = valid_scores[:-5] if len(valid_scores) > 5 else []
        
        if older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            
            if recent_avg > older_avg + 5:
                trend = "improving"
            elif recent_avg < older_avg - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "confidence": min(len(valid_scores) / 10.0, 1.0),
            "recent_average": sum(recent_scores) / len(recent_scores) if recent_scores else 0,
            "overall_average": sum(valid_scores) / len(valid_scores)
        }
    
    def _generate_effectiveness_recommendations(self, effectiveness_analysis: List[Dict[str, Any]], 
                                              trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on effectiveness analysis"""
        recommendations = []
        
        # Analyze trend
        if trends["trend"] == "declining" and trends["confidence"] > 0.5:
            recommendations.append("Effectiveness is declining - review allocation algorithms and parameters")
        
        # Analyze individual action performance
        low_performing_actions = [a for a in effectiveness_analysis 
                                if a.get("effectiveness_score", 0) < 30]
        
        if len(low_performing_actions) > len(effectiveness_analysis) * 0.3:
            recommendations.append("High number of low-performing actions - consider adjusting allocation thresholds")
        
        # Check for specific patterns
        action_types = {}
        for analysis in effectiveness_analysis:
            action_type = analysis.get("action_type", "unknown")
            if action_type not in action_types:
                action_types[action_type] = []
            if analysis.get("effectiveness_score") is not None:
                action_types[action_type].append(analysis["effectiveness_score"])
        
        for action_type, scores in action_types.items():
            if scores and sum(scores) / len(scores) < 40:
                recommendations.append(f"Low effectiveness for {action_type} actions - review implementation")
        
        return recommendations if recommendations else ["Allocation effectiveness appears satisfactory"]
    
    def _calculate_performance_metrics(self, spectrum_actions: List[OptimizationAction]) -> Dict[str, Any]:
        """Calculate performance metrics for spectrum actions"""
        if not spectrum_actions:
            return {}
        
        # Execution success rate
        completed_actions = [a for a in spectrum_actions if a.status == ActionStatus.COMPLETED]
        success_rate = (len(completed_actions) / len(spectrum_actions)) * 100
        
        # Average effectiveness
        effectiveness_scores = [a.effectiveness_score for a in spectrum_actions 
                              if a.effectiveness_score is not None]
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0
        
        # Action frequency
        time_span = (spectrum_actions[0].executed_at - spectrum_actions[-1].executed_at).total_seconds() / 3600
        action_frequency = len(spectrum_actions) / time_span if time_span > 0 else 0
        
        return {
            "total_actions": len(spectrum_actions),
            "completed_actions": len(completed_actions),
            "success_rate": round(success_rate, 2),
            "average_effectiveness": round(avg_effectiveness, 2),
            "actions_per_hour": round(action_frequency, 2),
            "time_span_hours": round(time_span, 2)
        }
    
    def _assess_load_balancing_sla_risk(self, selected_towers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess SLA risk for load balancing operations"""
        risk_assessment = {
            "overall_risk": "LOW",
            "risk_factors": [],
            "mitigation_strategies": []
        }
        
        high_risk_count = 0
        
        for tower in selected_towers:
            post_transfer_load = tower["post_transfer_load"]
            
            if post_transfer_load > 85:
                risk_assessment["risk_factors"].append({
                    "tower_id": tower["tower_id"],
                    "risk_type": "HIGH_POST_TRANSFER_LOAD",
                    "risk_level": "HIGH",
                    "details": f"Post-transfer load will be {post_transfer_load:.1f}%"
                })
                high_risk_count += 1
            elif post_transfer_load > 75:
                risk_assessment["risk_factors"].append({
                    "tower_id": tower["tower_id"],
                    "risk_type": "MEDIUM_POST_TRANSFER_LOAD",
                    "risk_level": "MEDIUM",
                    "details": f"Post-transfer load will be {post_transfer_load:.1f}%"
                })
        
        # Determine overall risk level
        if high_risk_count > 0:
            risk_assessment["overall_risk"] = "HIGH"
            risk_assessment["mitigation_strategies"].append("Reduce load transfer amounts")
            risk_assessment["mitigation_strategies"].append("Implement gradual load transfer")
        elif len(risk_assessment["risk_factors"]) > 0:
            risk_assessment["overall_risk"] = "MEDIUM"
            risk_assessment["mitigation_strategies"].append("Monitor closely during transfer")
        
        return risk_assessment
    
    def _assess_traffic_redirection_sla_risk(self, redirection_distribution: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess SLA risk for traffic redirection operations"""
        risk_assessment = {
            "overall_risk": "LOW",
            "risk_factors": [],
            "mitigation_strategies": []
        }
        
        high_risk_count = 0
        
        for redirection in redirection_distribution:
            post_redirect_load = redirection["estimated_post_redirect_load"]
            
            if post_redirect_load > 90:
                risk_assessment["risk_factors"].append({
                    "tower_id": redirection["target_tower_id"],
                    "risk_type": "HIGH_POST_REDIRECT_LOAD",
                    "risk_level": "HIGH",
                    "details": f"Post-redirection load will be {post_redirect_load:.1f}%"
                })
                high_risk_count += 1
            elif post_redirect_load > 80:
                risk_assessment["risk_factors"].append({
                    "tower_id": redirection["target_tower_id"],
                    "risk_type": "MEDIUM_POST_REDIRECT_LOAD",
                    "risk_level": "MEDIUM",
                    "details": f"Post-redirection load will be {post_redirect_load:.1f}%"
                })
        
        # Determine overall risk level
        if high_risk_count > 0:
            risk_assessment["overall_risk"] = "HIGH"
            risk_assessment["mitigation_strategies"].append("Reduce traffic redirection percentage")
            risk_assessment["mitigation_strategies"].append("Implement gradual traffic redirection")
        elif len(risk_assessment["risk_factors"]) > 0:
            risk_assessment["overall_risk"] = "MEDIUM"
            risk_assessment["mitigation_strategies"].append("Monitor closely during redirection")
        
        return risk_assessment
    
    def _calculate_service_availability(self, metrics: TowerMetrics) -> float:
        """Calculate service availability percentage"""
        # Simplified calculation based on error rate and system health
        base_availability = 100.0
        
        # Reduce availability based on error rate
        availability_reduction = metrics.error_rate * 0.1  # 1% error rate = 0.1% availability reduction
        
        # Reduce availability based on high resource utilization
        max_utilization = max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage)
        if max_utilization > 95:
            availability_reduction += (max_utilization - 95) * 0.2
        
        return max(base_availability - availability_reduction, 95.0)  # Minimum 95% availability
    
    def _check_data_speed_compliance(self, metrics: TowerMetrics) -> Dict[str, Any]:
        """Check data speed compliance (30-second tolerance for drops)"""
        # Simplified check based on bandwidth usage and signal strength
        bandwidth_impact = max(0, metrics.bandwidth_usage - 80) * 0.5  # Speed reduction factor
        signal_impact = max(0, -70 - metrics.signal_strength) * 2  # Signal strength impact
        
        speed_reduction_percentage = bandwidth_impact + signal_impact
        compliant = speed_reduction_percentage <= 20  # Allow up to 20% speed reduction
        
        return {
            "compliant": compliant,
            "speed_reduction_percentage": speed_reduction_percentage,
            "bandwidth_impact": bandwidth_impact,
            "signal_impact": signal_impact
        }
    
    def _calculate_call_drop_rate(self, metrics: TowerMetrics) -> float:
        """Calculate call drop rate percentage"""
        # Simplified calculation based on error rate and resource utilization
        base_drop_rate = metrics.error_rate * 0.01  # Convert error rate to drop rate
        
        # Increase drop rate based on high utilization
        max_utilization = max(metrics.cpu_utilization, metrics.memory_usage, metrics.bandwidth_usage)
        if max_utilization > 90:
            base_drop_rate += (max_utilization - 90) * 0.01
        
        return min(base_drop_rate, 5.0)  # Cap at 5% maximum


# MCP Server Implementation

server = Server("spectrum-allocation-mcp")

# Initialize spectrum manager
spectrum_manager = SpectrumManager()

@server.list_tools()
async def handle_list_tools() -> list[mcp_types.Tool]:
    """List available spectrum allocation tools"""
    return [
        mcp_types.Tool(
            name="analyze_spectrum_usage",
            description="Analyze current spectrum utilization for a specific tower",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "ID of the tower to analyze"
                    }
                },
                "required": ["tower_id"]
            }
        ),
        mcp_types.Tool(
            name="calculate_optimal_allocation",
            description="Calculate optimal bandwidth distribution across multiple towers",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tower IDs to optimize"
                    },
                    "target_improvement": {
                        "type": "number",
                        "description": "Target improvement percentage (default: 20.0)",
                        "default": 20.0
                    }
                },
                "required": ["tower_ids"]
            }
        ),
        mcp_types.Tool(
            name="execute_reallocation",
            description="Execute spectrum reallocation based on provided plan",
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_data": {
                        "type": "object",
                        "description": "Allocation plan data with actions to execute",
                        "properties": {
                            "plan_id": {"type": "string"},
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "description": "Individual reallocation action"
                                }
                            }
                        },
                        "required": ["actions"]
                    }
                },
                "required": ["plan_data"]
            }
        ),
        mcp_types.Tool(
            name="monitor_allocation_effectiveness",
            description="Monitor effectiveness of recent spectrum allocations",
            inputSchema={
                "type": "object",
                "properties": {
                    "hours_back": {
                        "type": "integer",
                        "description": "Hours to look back for analysis (default: 24)",
                        "default": 24
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="execute_dynamic_reallocation",
            description="Execute dynamic bandwidth reallocation using advanced algorithms",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tower IDs for dynamic reallocation"
                    },
                    "algorithm": {
                        "type": "string",
                        "description": "Reallocation algorithm to use",
                        "enum": ["load_balancing", "congestion_relief", "efficiency_optimization", "predictive_allocation"],
                        "default": "load_balancing"
                    }
                },
                "required": ["tower_ids"]
            }
        ),
        mcp_types.Tool(
            name="monitor_reallocation_effectiveness_advanced",
            description="Advanced monitoring of reallocation effectiveness with detailed analytics",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_window_hours": {
                        "type": "integer",
                        "description": "Time window in hours for analysis (default: 24)",
                        "default": 24
                    }
                }
            }
        ),
        mcp_types.Tool(
            name="optimize_spectrum_efficiency",
            description="Optimize spectrum efficiency across towers using advanced algorithms",
            inputSchema={
                "type": "object",
                "properties": {
                    "tower_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tower IDs for efficiency optimization"
                    }
                },
                "required": ["tower_ids"]
            }
        ),
        mcp_types.Tool(
            name="execute_cross_tower_load_balancing",
            description="Execute cross-tower load balancing for congestion relief (Requirement 3.4)",
            inputSchema={
                "type": "object",
                "properties": {
                    "congested_tower_id": {
                        "type": "string",
                        "description": "ID of the congested tower requiring load balancing"
                    },
                    "neighbor_radius_km": {
                        "type": "number",
                        "description": "Radius in kilometers to search for neighboring towers (default: 5.0)",
                        "default": 5.0
                    }
                },
                "required": ["congested_tower_id"]
            }
        ),
        mcp_types.Tool(
            name="execute_traffic_redirection",
            description="Execute traffic redirection for congestion relief with SLA monitoring",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_tower_id": {
                        "type": "string",
                        "description": "ID of the source tower to redirect traffic from"
                    },
                    "target_tower_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target tower IDs to redirect traffic to"
                    },
                    "traffic_percentage": {
                        "type": "number",
                        "description": "Percentage of traffic to redirect (10-50%, default: 30.0)",
                        "minimum": 10.0,
                        "maximum": 50.0,
                        "default": 30.0
                    }
                },
                "required": ["source_tower_id", "target_tower_ids"]
            }
        ),
        mcp_types.Tool(
            name="monitor_sla_compliance_during_optimization",
            description="Monitor SLA compliance during optimization activities (Requirements 6.1, 6.2)",
            inputSchema={
                "type": "object",
                "properties": {
                    "optimization_id": {
                        "type": "string",
                        "description": "ID of the optimization action to monitor"
                    },
                    "monitoring_duration_minutes": {
                        "type": "integer",
                        "description": "Duration in minutes to monitor SLA compliance (default: 10)",
                        "default": 10,
                        "minimum": 5,
                        "maximum": 60
                    }
                },
                "required": ["optimization_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """Handle tool calls for spectrum allocation operations"""
    try:
        if name == "analyze_spectrum_usage":
            tower_id = arguments.get("tower_id")
            result = spectrum_manager.analyze_spectrum_usage(tower_id)
            
        elif name == "calculate_optimal_allocation":
            tower_ids = arguments.get("tower_ids", [])
            target_improvement = arguments.get("target_improvement", 20.0)
            result = spectrum_manager.calculate_optimal_allocation(tower_ids, target_improvement)
            
        elif name == "execute_reallocation":
            plan_data = arguments.get("plan_data", {})
            result = spectrum_manager.execute_reallocation(plan_data)
            
        elif name == "monitor_allocation_effectiveness":
            hours_back = arguments.get("hours_back", 24)
            result = spectrum_manager.monitor_allocation_effectiveness(hours_back)
            
        elif name == "execute_dynamic_reallocation":
            tower_ids = arguments.get("tower_ids", [])
            algorithm = arguments.get("algorithm", "load_balancing")
            result = spectrum_manager.execute_dynamic_reallocation(tower_ids, algorithm)
            
        elif name == "monitor_reallocation_effectiveness_advanced":
            time_window_hours = arguments.get("time_window_hours", 24)
            result = spectrum_manager.monitor_reallocation_effectiveness_advanced(time_window_hours)
            
        elif name == "optimize_spectrum_efficiency":
            tower_ids = arguments.get("tower_ids", [])
            result = spectrum_manager.optimize_spectrum_efficiency(tower_ids)
            
        elif name == "execute_cross_tower_load_balancing":
            congested_tower_id = arguments.get("congested_tower_id")
            neighbor_radius_km = arguments.get("neighbor_radius_km", 5.0)
            result = spectrum_manager.execute_cross_tower_load_balancing(congested_tower_id, neighbor_radius_km)
            
        elif name == "execute_traffic_redirection":
            source_tower_id = arguments.get("source_tower_id")
            target_tower_ids = arguments.get("target_tower_ids", [])
            traffic_percentage = arguments.get("traffic_percentage", 30.0)
            result = spectrum_manager.execute_traffic_redirection(source_tower_id, target_tower_ids, traffic_percentage)
            
        elif name == "monitor_sla_compliance_during_optimization":
            optimization_id = arguments.get("optimization_id")
            monitoring_duration_minutes = arguments.get("monitoring_duration_minutes", 10)
            result = await spectrum_manager.monitor_sla_compliance_during_optimization(optimization_id, monitoring_duration_minutes)
            
        else:
            result = {"success": False, "message": f"Unknown tool: {name}"}
        
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        logging.error(f"Error handling tool call {name}: {e}")
        error_result = {
            "success": False,
            "message": f"Error executing {name}: {str(e)}",
            "tool": name,
            "arguments": arguments
        }
        return [mcp_types.TextContent(
            type="text",
            text=json.dumps(error_result, indent=2)
        )]

async def main():
    """Main entry point for the spectrum allocation MCP server"""
    # Setup logging
    logging.info("Starting Spectrum Allocation MCP Server...")
    
    # Initialize database connection
    try:
        spectrum_manager.db_manager.get_connection().close()
        logging.info("Database connection verified")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return
    
    # Run the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="spectrum-allocation-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
            
