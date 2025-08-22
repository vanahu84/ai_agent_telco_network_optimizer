"""
Dynamic Bandwidth Reallocation Logic
Implements advanced algorithms for optimal bandwidth distribution and spectrum reallocation
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import math

from network_models import (
    SpectrumAllocation, TowerMetrics, Tower, SeverityLevel,
    OptimizationAction, OptimizationActionType, ActionStatus
)
from network_db_utils import NetworkDatabaseManager

class DynamicReallocationEngine:
    """Advanced engine for dynamic bandwidth reallocation"""
    
    def __init__(self, db_manager: NetworkDatabaseManager):
        self.db_manager = db_manager
        self.reallocation_algorithms = {
            'load_balancing': self._load_balancing_algorithm,
            'congestion_relief': self._congestion_relief_algorithm,
            'efficiency_optimization': self._efficiency_optimization_algorithm,
            'predictive_allocation': self._predictive_allocation_algorithm
        }
        
        # Algorithm parameters
        self.congestion_threshold = 80.0
        self.efficiency_threshold = 60.0
        self.load_balance_tolerance = 15.0  # Max load difference for balancing
        self.min_allocation_bandwidth = 10.0  # Minimum bandwidth per allocation
        self.max_allocation_bandwidth = 500.0  # Maximum bandwidth per allocation
        
    def calculate_optimal_distribution(self, tower_data: List[Dict[str, Any]], 
                                     algorithm: str = 'load_balancing') -> Dict[str, Any]:
        """Calculate optimal bandwidth distribution using specified algorithm"""
        try:
            if algorithm not in self.reallocation_algorithms:
                return {
                    "success": False,
                    "message": f"Unknown algorithm: {algorithm}",
                    "available_algorithms": list(self.reallocation_algorithms.keys())
                }
            
            # Apply the selected algorithm
            algorithm_func = self.reallocation_algorithms[algorithm]
            distribution_plan = algorithm_func(tower_data)
            
            # Validate and optimize the plan
            validated_plan = self._validate_distribution_plan(distribution_plan, tower_data)
            
            # Calculate expected improvements
            improvements = self._calculate_expected_improvements(validated_plan, tower_data)
            
            result = {
                "success": True,
                "algorithm_used": algorithm,
                "calculation_timestamp": datetime.now().isoformat(),
                "distribution_plan": validated_plan,
                "expected_improvements": improvements,
                "execution_metadata": {
                    "total_towers": len(tower_data),
                    "total_reallocations": len(validated_plan.get("reallocations", [])),
                    "estimated_execution_time_minutes": self._estimate_execution_time(validated_plan),
                    "complexity_score": self._calculate_complexity_score(validated_plan)
                }
            }
            
            logging.info(f"Calculated optimal distribution using {algorithm}: "
                        f"{len(validated_plan.get('reallocations', []))} reallocations planned")
            
            return result
            
        except Exception as e:
            logging.error(f"Error calculating optimal distribution: {e}")
            return {
                "success": False,
                "message": f"Error calculating optimal distribution: {str(e)}",
                "algorithm": algorithm
            }
    
    def execute_reallocation_plan(self, distribution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a bandwidth reallocation plan with rollback capability"""
        try:
            reallocations = distribution_plan.get("reallocations", [])
            if not reallocations:
                return {
                    "success": False,
                    "message": "No reallocations specified in plan"
                }
            
            execution_results = []
            rollback_actions = []
            successful_count = 0
            failed_count = 0
            
            # Execute reallocations in priority order
            sorted_reallocations = sorted(reallocations, 
                                        key=lambda x: self._get_priority_score(x), 
                                        reverse=True)
            
            for reallocation in sorted_reallocations:
                try:
                    # Store current state for potential rollback
                    rollback_info = self._capture_rollback_state(reallocation)
                    
                    # Execute the reallocation
                    execution_result = self._execute_single_reallocation(reallocation)
                    
                    if execution_result["success"]:
                        successful_count += 1
                        rollback_actions.append(rollback_info)
                        
                        # Log successful action
                        self._log_reallocation_action(reallocation, execution_result, "SUCCESS")
                        
                    else:
                        failed_count += 1
                        logging.warning(f"Reallocation failed: {execution_result.get('error', 'Unknown error')}")
                    
                    execution_results.append(execution_result)
                    
                except Exception as reallocation_error:
                    failed_count += 1
                    error_result = {
                        "success": False,
                        "reallocation": reallocation,
                        "error": str(reallocation_error)
                    }
                    execution_results.append(error_result)
                    logging.error(f"Reallocation execution error: {reallocation_error}")
            
            # Calculate overall success rate
            success_rate = (successful_count / len(reallocations)) * 100 if reallocations else 0
            
            # Determine if rollback is needed (if success rate is too low)
            rollback_performed = False
            if success_rate < 50 and rollback_actions:
                logging.warning(f"Low success rate ({success_rate:.1f}%), initiating rollback")
                rollback_result = self._perform_rollback(rollback_actions)
                rollback_performed = rollback_result["success"]
            
            result = {
                "success": successful_count > 0,
                "execution_timestamp": datetime.now().isoformat(),
                "execution_summary": {
                    "total_reallocations": len(reallocations),
                    "successful_reallocations": successful_count,
                    "failed_reallocations": failed_count,
                    "success_rate": round(success_rate, 2),
                    "rollback_performed": rollback_performed
                },
                "execution_results": execution_results,
                "post_execution_analysis": self._analyze_post_execution_state(distribution_plan)
            }
            
            logging.info(f"Executed reallocation plan: {successful_count}/{len(reallocations)} successful")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing reallocation plan: {e}")
            return {
                "success": False,
                "message": f"Error executing reallocation plan: {str(e)}"
            }
    
    def monitor_reallocation_effectiveness(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Monitor effectiveness of recent reallocations with detailed analysis"""
        try:
            # Get recent optimization actions
            recent_actions = self.db_manager.get_recent_optimization_actions(time_window_hours)
            reallocation_actions = [a for a in recent_actions 
                                  if a.action_type == OptimizationActionType.SPECTRUM_REALLOCATION]
            
            if not reallocation_actions:
                return {
                    "success": True,
                    "message": f"No reallocation actions found in the last {time_window_hours} hours",
                    "monitoring_period_hours": time_window_hours,
                    "total_actions": 0
                }
            
            # Analyze each reallocation action
            effectiveness_analysis = []
            performance_metrics = {
                "total_bandwidth_reallocated": 0,
                "average_effectiveness_score": 0,
                "improvement_distribution": {"high": 0, "medium": 0, "low": 0},
                "algorithm_performance": {}
            }
            
            for action in reallocation_actions:
                analysis = self._analyze_reallocation_effectiveness(action)
                effectiveness_analysis.append(analysis)
                
                # Update performance metrics
                if analysis.get("bandwidth_reallocated"):
                    performance_metrics["total_bandwidth_reallocated"] += analysis["bandwidth_reallocated"]
                
                effectiveness_score = analysis.get("effectiveness_score", 0)
                if effectiveness_score >= 70:
                    performance_metrics["improvement_distribution"]["high"] += 1
                elif effectiveness_score >= 40:
                    performance_metrics["improvement_distribution"]["medium"] += 1
                else:
                    performance_metrics["improvement_distribution"]["low"] += 1
            
            # Calculate average effectiveness
            valid_scores = [a["effectiveness_score"] for a in effectiveness_analysis 
                          if a.get("effectiveness_score") is not None]
            performance_metrics["average_effectiveness_score"] = (
                sum(valid_scores) / len(valid_scores) if valid_scores else 0
            )
            
            # Identify trends and patterns
            trends = self._identify_reallocation_trends(effectiveness_analysis)
            
            # Generate improvement recommendations
            recommendations = self._generate_reallocation_recommendations(effectiveness_analysis, trends)
            
            result = {
                "success": True,
                "monitoring_timestamp": datetime.now().isoformat(),
                "monitoring_period_hours": time_window_hours,
                "effectiveness_summary": {
                    "total_actions_analyzed": len(reallocation_actions),
                    "actions_with_valid_scores": len(valid_scores),
                    **performance_metrics
                },
                "detailed_analysis": effectiveness_analysis,
                "trends": trends,
                "recommendations": recommendations,
                "network_impact_assessment": self._assess_network_impact(reallocation_actions)
            }
            
            logging.info(f"Monitored reallocation effectiveness: {len(reallocation_actions)} actions, "
                        f"average effectiveness {performance_metrics['average_effectiveness_score']:.1f}%")
            
            return result
            
        except Exception as e:
            logging.error(f"Error monitoring reallocation effectiveness: {e}")
            return {
                "success": False,
                "message": f"Error monitoring reallocation effectiveness: {str(e)}",
                "monitoring_period_hours": time_window_hours
            }
    
    # Algorithm implementations
    
    def _load_balancing_algorithm(self, tower_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load balancing algorithm for even distribution of network load"""
        reallocations = []
        
        # Calculate average load across all towers
        total_load = sum(data["current_load"] for data in tower_data)
        average_load = total_load / len(tower_data) if tower_data else 0
        
        # Identify overloaded and underloaded towers
        overloaded_towers = [data for data in tower_data 
                           if data["current_load"] > average_load + self.load_balance_tolerance]
        underloaded_towers = [data for data in tower_data 
                            if data["current_load"] < average_load - self.load_balance_tolerance]
        
        # Create reallocation pairs
        for overloaded in overloaded_towers:
            for underloaded in underloaded_towers:
                if overloaded["current_load"] <= average_load + self.load_balance_tolerance:
                    break  # This tower is now balanced
                
                # Calculate bandwidth to transfer
                load_difference = overloaded["current_load"] - average_load
                transfer_bandwidth = min(load_difference * 2, 50.0)  # Heuristic conversion
                
                # Check if underloaded tower can accept the bandwidth
                underloaded_capacity = (average_load + self.load_balance_tolerance) - underloaded["current_load"]
                actual_transfer = min(transfer_bandwidth, underloaded_capacity * 2)
                
                if actual_transfer >= self.min_allocation_bandwidth:
                    reallocation = {
                        "type": "load_balance_transfer",
                        "source_tower": overloaded["tower"].id,
                        "target_tower": underloaded["tower"].id,
                        "bandwidth_amount": actual_transfer,
                        "priority": "MEDIUM",
                        "expected_load_reduction": actual_transfer / 2,
                        "algorithm": "load_balancing"
                    }
                    reallocations.append(reallocation)
                    
                    # Update loads for next iteration
                    overloaded["current_load"] -= actual_transfer / 2
                    underloaded["current_load"] += actual_transfer / 2
        
        return {
            "algorithm": "load_balancing",
            "reallocations": reallocations,
            "summary": {
                "target_average_load": average_load,
                "overloaded_towers_initial": len([d for d in tower_data 
                                                if d["current_load"] > average_load + self.load_balance_tolerance]),
                "reallocations_planned": len(reallocations)
            }
        }
    
    def _congestion_relief_algorithm(self, tower_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Congestion relief algorithm focusing on high-priority congested towers"""
        reallocations = []
        
        # Sort towers by congestion severity and load
        congested_towers = sorted(
            [data for data in tower_data if data["current_load"] > self.congestion_threshold],
            key=lambda x: (x["congestion_severity"].value == "HIGH", x["current_load"]),
            reverse=True
        )
        
        # Find available capacity in non-congested towers
        available_towers = [data for data in tower_data 
                          if data["current_load"] < self.congestion_threshold - 10]
        
        for congested in congested_towers:
            congestion_level = congested["current_load"] - self.congestion_threshold
            relief_needed = congestion_level * 1.5  # Add buffer
            
            # Distribute relief across available towers
            remaining_relief = relief_needed
            
            for available in available_towers:
                if remaining_relief <= 0:
                    break
                
                # Calculate how much this tower can accept
                available_capacity = (self.congestion_threshold - 10) - available["current_load"]
                transfer_amount = min(remaining_relief, available_capacity, 100.0)  # Max 100 MHz per transfer
                
                if transfer_amount >= self.min_allocation_bandwidth:
                    reallocation = {
                        "type": "congestion_relief",
                        "source_tower": congested["tower"].id,
                        "target_tower": available["tower"].id,
                        "bandwidth_amount": transfer_amount,
                        "priority": "HIGH" if congested["congestion_severity"] == SeverityLevel.HIGH else "MEDIUM",
                        "expected_congestion_reduction": transfer_amount / 1.5,
                        "algorithm": "congestion_relief"
                    }
                    reallocations.append(reallocation)
                    
                    remaining_relief -= transfer_amount
                    available["current_load"] += transfer_amount / 1.5  # Update for next iteration
        
        return {
            "algorithm": "congestion_relief",
            "reallocations": reallocations,
            "summary": {
                "congested_towers_processed": len(congested_towers),
                "available_towers_used": len([t for t in available_towers 
                                            if any(r["target_tower"] == t["tower"].id for r in reallocations)]),
                "total_relief_planned": sum(r["bandwidth_amount"] for r in reallocations)
            }
        }
    
    def _efficiency_optimization_algorithm(self, tower_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Efficiency optimization algorithm to improve spectrum utilization"""
        reallocations = []
        
        # Calculate efficiency scores for all towers
        for data in tower_data:
            allocations = data.get("allocations", [])
            metrics = data["metrics"]
            
            if allocations:
                efficiency_score = self._calculate_tower_efficiency(allocations, metrics)
                data["efficiency_score"] = efficiency_score
            else:
                data["efficiency_score"] = 0
        
        # Identify inefficient towers
        inefficient_towers = [data for data in tower_data 
                            if data["efficiency_score"] < self.efficiency_threshold]
        
        for tower_data_item in inefficient_towers:
            current_allocations = tower_data_item.get("allocations", [])
            metrics = tower_data_item["metrics"]
            
            # Analyze current allocation efficiency
            optimization_actions = self._analyze_allocation_efficiency(current_allocations, metrics)
            
            for action in optimization_actions:
                if action["type"] == "reallocate_band":
                    reallocation = {
                        "type": "efficiency_optimization",
                        "tower_id": tower_data_item["tower"].id,
                        "current_band": action["from_band"],
                        "target_band": action["to_band"],
                        "bandwidth_amount": action["bandwidth"],
                        "priority": "MEDIUM",
                        "expected_efficiency_gain": action["efficiency_gain"],
                        "algorithm": "efficiency_optimization"
                    }
                    reallocations.append(reallocation)
                
                elif action["type"] == "consolidate_allocations":
                    reallocation = {
                        "type": "consolidation",
                        "tower_id": tower_data_item["tower"].id,
                        "consolidate_bands": action["bands"],
                        "target_band": action["target_band"],
                        "total_bandwidth": action["total_bandwidth"],
                        "priority": "LOW",
                        "expected_efficiency_gain": action["efficiency_gain"],
                        "algorithm": "efficiency_optimization"
                    }
                    reallocations.append(reallocation)
        
        return {
            "algorithm": "efficiency_optimization",
            "reallocations": reallocations,
            "summary": {
                "inefficient_towers_identified": len(inefficient_towers),
                "optimization_actions_planned": len(reallocations),
                "average_efficiency_before": sum(d["efficiency_score"] for d in tower_data) / len(tower_data),
                "expected_efficiency_improvement": sum(r.get("expected_efficiency_gain", 0) for r in reallocations)
            }
        }
    
    def _predictive_allocation_algorithm(self, tower_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predictive allocation algorithm based on historical patterns and trends"""
        reallocations = []
        
        # Analyze historical patterns for each tower
        for data in tower_data:
            tower_id = data["tower"].id
            
            # Get historical metrics (simplified - would use more sophisticated prediction in real implementation)
            historical_analysis = self._analyze_historical_patterns(tower_id)
            
            if historical_analysis["trend"] == "increasing" and historical_analysis["confidence"] > 0.7:
                # Predict future congestion and preemptively allocate bandwidth
                predicted_increase = historical_analysis["predicted_load_increase"]
                
                if data["current_load"] + predicted_increase > self.congestion_threshold:
                    # Find source towers with decreasing trends
                    source_candidates = [other for other in tower_data 
                                       if other["tower"].id != tower_id and 
                                       self._analyze_historical_patterns(other["tower"].id)["trend"] == "decreasing"]
                    
                    if source_candidates:
                        source_tower = min(source_candidates, key=lambda x: x["current_load"])
                        transfer_amount = min(predicted_increase * 2, 75.0)  # Preemptive allocation
                        
                        reallocation = {
                            "type": "predictive_allocation",
                            "source_tower": source_tower["tower"].id,
                            "target_tower": tower_id,
                            "bandwidth_amount": transfer_amount,
                            "priority": "MEDIUM",
                            "prediction_confidence": historical_analysis["confidence"],
                            "predicted_load_increase": predicted_increase,
                            "algorithm": "predictive_allocation"
                        }
                        reallocations.append(reallocation)
        
        return {
            "algorithm": "predictive_allocation",
            "reallocations": reallocations,
            "summary": {
                "towers_analyzed": len(tower_data),
                "predictive_reallocations": len(reallocations),
                "average_prediction_confidence": sum(r["prediction_confidence"] for r in reallocations) / len(reallocations) if reallocations else 0
            }
        }    
 
   # Helper methods for reallocation execution and analysis
    
    def _validate_distribution_plan(self, plan: Dict[str, Any], tower_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and optimize distribution plan"""
        reallocations = plan.get("reallocations", [])
        validated_reallocations = []
        
        for reallocation in reallocations:
            # Validate bandwidth amounts
            bandwidth = reallocation.get("bandwidth_amount", 0)
            if self.min_allocation_bandwidth <= bandwidth <= self.max_allocation_bandwidth:
                
                # Validate tower capacity
                if self._validate_tower_capacity(reallocation, tower_data):
                    validated_reallocations.append(reallocation)
                else:
                    logging.warning(f"Reallocation failed capacity validation: {reallocation}")
            else:
                logging.warning(f"Reallocation bandwidth out of range: {bandwidth} MHz")
        
        validated_plan = plan.copy()
        validated_plan["reallocations"] = validated_reallocations
        validated_plan["validation_summary"] = {
            "original_count": len(reallocations),
            "validated_count": len(validated_reallocations),
            "validation_rate": (len(validated_reallocations) / len(reallocations)) * 100 if reallocations else 100
        }
        
        return validated_plan
    
    def _validate_tower_capacity(self, reallocation: Dict[str, Any], tower_data: List[Dict[str, Any]]) -> bool:
        """Validate that towers have capacity for the reallocation"""
        source_tower_id = reallocation.get("source_tower")
        target_tower_id = reallocation.get("target_tower")
        bandwidth_amount = reallocation.get("bandwidth_amount", 0)
        
        # Find tower data
        source_data = next((d for d in tower_data if d["tower"].id == source_tower_id), None)
        target_data = next((d for d in tower_data if d["tower"].id == target_tower_id), None)
        
        if not source_data or not target_data:
            return False
        
        # Check if source tower has enough bandwidth to give
        source_allocations = source_data.get("allocations", [])
        total_source_bandwidth = sum(a.allocated_bandwidth for a in source_allocations)
        
        if total_source_bandwidth < bandwidth_amount:
            return False
        
        # Check if target tower can accept the bandwidth
        target_current_load = target_data["current_load"]
        estimated_load_increase = bandwidth_amount / 2  # Heuristic conversion
        
        if target_current_load + estimated_load_increase > 95:  # Don't overload target
            return False
        
        return True
    
    def _calculate_expected_improvements(self, plan: Dict[str, Any], tower_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate expected improvements from the distribution plan"""
        reallocations = plan.get("reallocations", [])
        
        improvements = {
            "load_reduction": {},
            "congestion_relief": {},
            "efficiency_gains": {},
            "overall_network_improvement": 0
        }
        
        total_improvement = 0
        
        for reallocation in reallocations:
            reallocation_type = reallocation.get("type", "unknown")
            
            if reallocation_type == "load_balance_transfer":
                source_id = reallocation["source_tower"]
                target_id = reallocation["target_tower"]
                load_reduction = reallocation.get("expected_load_reduction", 0)
                
                improvements["load_reduction"][source_id] = improvements["load_reduction"].get(source_id, 0) + load_reduction
                total_improvement += load_reduction * 0.5  # Weight for load balancing
                
            elif reallocation_type == "congestion_relief":
                source_id = reallocation["source_tower"]
                congestion_reduction = reallocation.get("expected_congestion_reduction", 0)
                
                improvements["congestion_relief"][source_id] = improvements["congestion_relief"].get(source_id, 0) + congestion_reduction
                total_improvement += congestion_reduction * 1.0  # Higher weight for congestion relief
                
            elif reallocation_type == "efficiency_optimization":
                tower_id = reallocation["tower_id"]
                efficiency_gain = reallocation.get("expected_efficiency_gain", 0)
                
                improvements["efficiency_gains"][tower_id] = improvements["efficiency_gains"].get(tower_id, 0) + efficiency_gain
                total_improvement += efficiency_gain * 0.3  # Lower weight for efficiency
        
        improvements["overall_network_improvement"] = min(total_improvement, 100.0)
        
        return improvements
    
    def _estimate_execution_time(self, plan: Dict[str, Any]) -> int:
        """Estimate execution time for the distribution plan"""
        reallocations = plan.get("reallocations", [])
        
        # Base time per reallocation type
        time_estimates = {
            "load_balance_transfer": 3,  # 3 minutes
            "congestion_relief": 2,      # 2 minutes (higher priority, faster execution)
            "efficiency_optimization": 4, # 4 minutes (more complex)
            "predictive_allocation": 3,   # 3 minutes
            "consolidation": 5           # 5 minutes (most complex)
        }
        
        total_time = 0
        for reallocation in reallocations:
            reallocation_type = reallocation.get("type", "load_balance_transfer")
            total_time += time_estimates.get(reallocation_type, 3)
        
        # Add overhead time
        overhead = max(len(reallocations) * 0.5, 2)  # Minimum 2 minutes overhead
        
        return int(total_time + overhead)
    
    def _calculate_complexity_score(self, plan: Dict[str, Any]) -> float:
        """Calculate complexity score for the distribution plan"""
        reallocations = plan.get("reallocations", [])
        
        if not reallocations:
            return 0.0
        
        # Complexity factors
        complexity_weights = {
            "load_balance_transfer": 1.0,
            "congestion_relief": 1.5,
            "efficiency_optimization": 2.0,
            "predictive_allocation": 2.5,
            "consolidation": 3.0
        }
        
        total_complexity = 0
        for reallocation in reallocations:
            reallocation_type = reallocation.get("type", "load_balance_transfer")
            weight = complexity_weights.get(reallocation_type, 1.0)
            
            # Additional complexity for cross-tower operations
            if "source_tower" in reallocation and "target_tower" in reallocation:
                weight *= 1.2
            
            total_complexity += weight
        
        # Normalize to 0-100 scale
        max_possible_complexity = len(reallocations) * 3.0 * 1.2
        complexity_score = (total_complexity / max_possible_complexity) * 100 if max_possible_complexity > 0 else 0
        
        return min(100.0, complexity_score)
    
    def _get_priority_score(self, reallocation: Dict[str, Any]) -> float:
        """Calculate priority score for reallocation ordering"""
        priority_map = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        base_priority = priority_map.get(reallocation.get("priority", "MEDIUM"), 2.0)
        
        # Boost priority for congestion relief
        if reallocation.get("type") == "congestion_relief":
            base_priority *= 1.5
        
        # Boost priority for high bandwidth amounts
        bandwidth = reallocation.get("bandwidth_amount", 0)
        if bandwidth > 100:
            base_priority *= 1.2
        
        return base_priority
    
    def _capture_rollback_state(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Capture current state for potential rollback"""
        rollback_info = {
            "reallocation": reallocation,
            "timestamp": datetime.now().isoformat(),
            "original_allocations": {}
        }
        
        # Capture affected towers' current allocations
        affected_towers = []
        if "source_tower" in reallocation:
            affected_towers.append(reallocation["source_tower"])
        if "target_tower" in reallocation:
            affected_towers.append(reallocation["target_tower"])
        if "tower_id" in reallocation:
            affected_towers.append(reallocation["tower_id"])
        
        for tower_id in affected_towers:
            current_allocations = self._get_current_allocations(tower_id)
            rollback_info["original_allocations"][tower_id] = [
                {
                    "tower_id": a.tower_id,
                    "frequency_band": a.frequency_band,
                    "allocated_bandwidth": a.allocated_bandwidth,
                    "utilization_percentage": a.utilization_percentage,
                    "allocation_timestamp": a.allocation_timestamp.isoformat(),
                    "expires_at": a.expires_at.isoformat() if a.expires_at else None
                }
                for a in current_allocations
            ]
        
        return rollback_info
    
    def _execute_single_reallocation(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reallocation action"""
        try:
            reallocation_type = reallocation.get("type", "load_balance_transfer")
            
            if reallocation_type == "load_balance_transfer":
                return self._execute_load_balance_transfer(reallocation)
            elif reallocation_type == "congestion_relief":
                return self._execute_congestion_relief(reallocation)
            elif reallocation_type == "efficiency_optimization":
                return self._execute_efficiency_optimization(reallocation)
            elif reallocation_type == "predictive_allocation":
                return self._execute_predictive_allocation(reallocation)
            elif reallocation_type == "consolidation":
                return self._execute_consolidation(reallocation)
            else:
                return {
                    "success": False,
                    "reallocation": reallocation,
                    "error": f"Unknown reallocation type: {reallocation_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "reallocation": reallocation,
                "error": str(e)
            }
        
        # Complexity factors
        complexity_weights = {
            "load_balance_transfer": 1.0,
            "congestion_relief": 1.5,
            "efficiency_optimization": 2.0,
            "predictive_allocation": 2.5,
            "consolidation": 3.0
        }
        
        total_complexity = 0
        for reallocation in reallocations:
            reallocation_type = reallocation.get("type", "load_balance_transfer")
            bandwidth = reallocation.get("bandwidth_amount", 0)
            
            base_complexity = complexity_weights.get(reallocation_type, 1.0)
            bandwidth_factor = min(bandwidth / 100.0, 2.0)  # Normalize bandwidth impact
            
            total_complexity += base_complexity * bandwidth_factor
        
        # Normalize to 0-100 scale
        normalized_complexity = min((total_complexity / len(reallocations)) * 20, 100)
        
        return round(normalized_complexity, 2)
    
    def _get_priority_score(self, reallocation: Dict[str, Any]) -> int:
        """Get priority score for reallocation ordering"""
        priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        priority = reallocation.get("priority", "MEDIUM")
        
        base_score = priority_map.get(priority, 2)
        
        # Adjust based on reallocation type
        type_adjustments = {
            "congestion_relief": 2,
            "load_balance_transfer": 1,
            "predictive_allocation": 1,
            "efficiency_optimization": 0,
            "consolidation": -1
        }
        
        reallocation_type = reallocation.get("type", "load_balance_transfer")
        adjustment = type_adjustments.get(reallocation_type, 0)
        
        return base_score + adjustment
    
    def _capture_rollback_state(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Capture current state for potential rollback"""
        rollback_info = {
            "reallocation": reallocation,
            "timestamp": datetime.now().isoformat(),
            "original_allocations": {}
        }
        
        # Capture current allocations for affected towers
        affected_towers = []
        if "source_tower" in reallocation:
            affected_towers.append(reallocation["source_tower"])
        if "target_tower" in reallocation:
            affected_towers.append(reallocation["target_tower"])
        if "tower_id" in reallocation:
            affected_towers.append(reallocation["tower_id"])
        
        for tower_id in affected_towers:
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM spectrum_allocations 
                        WHERE tower_id = ? AND (expires_at IS NULL OR expires_at > ?)
                    """, (tower_id, datetime.now().isoformat()))
                    
                    allocations = cursor.fetchall()
                    rollback_info["original_allocations"][tower_id] = [dict(row) for row in allocations]
                    
            except Exception as e:
                logging.error(f"Error capturing rollback state for {tower_id}: {e}")
        
        return rollback_info
    
    def _execute_single_reallocation(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reallocation action"""
        try:
            reallocation_type = reallocation.get("type")
            
            if reallocation_type == "load_balance_transfer":
                return self._execute_load_balance_transfer(reallocation)
            elif reallocation_type == "congestion_relief":
                return self._execute_congestion_relief(reallocation)
            elif reallocation_type == "efficiency_optimization":
                return self._execute_efficiency_optimization(reallocation)
            elif reallocation_type == "predictive_allocation":
                return self._execute_predictive_allocation(reallocation)
            elif reallocation_type == "consolidation":
                return self._execute_consolidation(reallocation)
            else:
                return {
                    "success": False,
                    "reallocation": reallocation,
                    "error": f"Unknown reallocation type: {reallocation_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "reallocation": reallocation,
                "error": str(e)
            }
    
    def _execute_load_balance_transfer(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute load balance transfer between towers"""
        source_tower = reallocation["source_tower"]
        target_tower = reallocation["target_tower"]
        bandwidth_amount = reallocation["bandwidth_amount"]
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Reduce bandwidth from source tower
                cursor.execute("""
                    UPDATE spectrum_allocations 
                    SET allocated_bandwidth = allocated_bandwidth - ?
                    WHERE tower_id = ? AND allocated_bandwidth >= ?
                    ORDER BY allocated_bandwidth DESC
                    LIMIT 1
                """, (bandwidth_amount, source_tower, bandwidth_amount))
                
                if cursor.rowcount == 0:
                    # Create new allocation reduction record
                    cursor.execute("""
                        INSERT INTO spectrum_allocations 
                        (tower_id, frequency_band, allocated_bandwidth, utilization_percentage, 
                         allocation_timestamp, expires_at)
                        VALUES (?, 'mid_band', ?, 0, ?, ?)
                    """, (source_tower, -bandwidth_amount, datetime.now().isoformat(),
                          (datetime.now() + timedelta(hours=1)).isoformat()))
                
                # Add bandwidth to target tower
                cursor.execute("""
                    INSERT INTO spectrum_allocations 
                    (tower_id, frequency_band, allocated_bandwidth, utilization_percentage, 
                     allocation_timestamp, expires_at)
                    VALUES (?, 'mid_band', ?, 0, ?, ?)
                """, (target_tower, bandwidth_amount, datetime.now().isoformat(),
                      (datetime.now() + timedelta(hours=1)).isoformat()))
                
                conn.commit()
                
                return {
                    "success": True,
                    "reallocation": reallocation,
                    "bandwidth_transferred": bandwidth_amount,
                    "source_tower": source_tower,
                    "target_tower": target_tower,
                    "execution_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "reallocation": reallocation,
                "error": f"Database error during load balance transfer: {str(e)}"
            }
    
    def _execute_congestion_relief(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute congestion relief reallocation"""
        # Similar to load balance transfer but with higher priority handling
        return self._execute_load_balance_transfer(reallocation)
    
    def _execute_efficiency_optimization(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute efficiency optimization reallocation"""
        tower_id = reallocation["tower_id"]
        
        if "current_band" in reallocation and "target_band" in reallocation:
            # Band reallocation
            current_band = reallocation["current_band"]
            target_band = reallocation["target_band"]
            bandwidth_amount = reallocation["bandwidth_amount"]
            
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Update existing allocation to new band
                    cursor.execute("""
                        UPDATE spectrum_allocations 
                        SET frequency_band = ?, allocation_timestamp = ?
                        WHERE tower_id = ? AND frequency_band = ? AND allocated_bandwidth = ?
                        LIMIT 1
                    """, (target_band, datetime.now().isoformat(), tower_id, current_band, bandwidth_amount))
                    
                    conn.commit()
                    
                    return {
                        "success": True,
                        "reallocation": reallocation,
                        "tower_id": tower_id,
                        "band_change": f"{current_band} -> {target_band}",
                        "bandwidth_amount": bandwidth_amount,
                        "execution_timestamp": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "reallocation": reallocation,
                    "error": f"Database error during efficiency optimization: {str(e)}"
                }
        
        return {
            "success": False,
            "reallocation": reallocation,
            "error": "Invalid efficiency optimization parameters"
        }
    
    def _execute_predictive_allocation(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute predictive allocation reallocation"""
        # Similar to load balance transfer but marked as predictive
        result = self._execute_load_balance_transfer(reallocation)
        if result["success"]:
            result["prediction_confidence"] = reallocation.get("prediction_confidence", 0)
            result["predicted_load_increase"] = reallocation.get("predicted_load_increase", 0)
        return result
    
    def _execute_consolidation(self, reallocation: Dict[str, Any]) -> Dict[str, Any]:
        """Execute allocation consolidation"""
        tower_id = reallocation["tower_id"]
        consolidate_bands = reallocation.get("consolidate_bands", [])
        target_band = reallocation["target_band"]
        total_bandwidth = reallocation["total_bandwidth"]
        
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Remove old allocations
                for band in consolidate_bands:
                    cursor.execute("""
                        DELETE FROM spectrum_allocations 
                        WHERE tower_id = ? AND frequency_band = ?
                    """, (tower_id, band))
                
                # Create consolidated allocation
                cursor.execute("""
                    INSERT INTO spectrum_allocations 
                    (tower_id, frequency_band, allocated_bandwidth, utilization_percentage, 
                     allocation_timestamp, expires_at)
                    VALUES (?, ?, ?, 0, ?, ?)
                """, (tower_id, target_band, total_bandwidth, datetime.now().isoformat(),
                      (datetime.now() + timedelta(hours=2)).isoformat()))
                
                conn.commit()
                
                return {
                    "success": True,
                    "reallocation": reallocation,
                    "tower_id": tower_id,
                    "consolidated_bands": consolidate_bands,
                    "target_band": target_band,
                    "total_bandwidth": total_bandwidth,
                    "execution_timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "success": False,
                "reallocation": reallocation,
                "error": f"Database error during consolidation: {str(e)}"
            }
    
    def _log_reallocation_action(self, reallocation: Dict[str, Any], result: Dict[str, Any], status: str):
        """Log reallocation action to database"""
        try:
            action = OptimizationAction(
                action_type=OptimizationActionType.SPECTRUM_REALLOCATION,
                tower_ids=[reallocation.get("source_tower", reallocation.get("tower_id", ""))],
                parameters=reallocation,
                executed_at=datetime.now(),
                effectiveness_score=result.get("effectiveness_score"),
                status=ActionStatus.COMPLETED if status == "SUCCESS" else ActionStatus.FAILED
            )
            
            self.db_manager.insert_optimization_action(action)
            
        except Exception as e:
            logging.error(f"Error logging reallocation action: {e}")
    
    def _perform_rollback(self, rollback_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform rollback of failed reallocations"""
        try:
            rollback_count = 0
            
            for rollback_info in reversed(rollback_actions):  # Reverse order for rollback
                try:
                    original_allocations = rollback_info["original_allocations"]
                    
                    with self.db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        
                        for tower_id, allocations in original_allocations.items():
                            # Clear current allocations
                            cursor.execute("DELETE FROM spectrum_allocations WHERE tower_id = ?", (tower_id,))
                            
                            # Restore original allocations
                            for allocation in allocations:
                                cursor.execute("""
                                    INSERT INTO spectrum_allocations 
                                    (tower_id, frequency_band, allocated_bandwidth, utilization_percentage, 
                                     allocation_timestamp, expires_at)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    allocation["tower_id"], allocation["frequency_band"],
                                    allocation["allocated_bandwidth"], allocation["utilization_percentage"],
                                    allocation["allocation_timestamp"], allocation["expires_at"]
                                ))
                        
                        conn.commit()
                        rollback_count += 1
                        
                except Exception as rollback_error:
                    logging.error(f"Error during individual rollback: {rollback_error}")
            
            return {
                "success": rollback_count > 0,
                "rollback_count": rollback_count,
                "total_attempted": len(rollback_actions)
            }
            
        except Exception as e:
            logging.error(f"Error during rollback: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_post_execution_state(self, distribution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze network state after reallocation execution"""
        reallocations = distribution_plan.get("reallocations", [])
        affected_towers = set()
        
        for reallocation in reallocations:
            if "source_tower" in reallocation:
                affected_towers.add(reallocation["source_tower"])
            if "target_tower" in reallocation:
                affected_towers.add(reallocation["target_tower"])
            if "tower_id" in reallocation:
                affected_towers.add(reallocation["tower_id"])
        
        post_analysis = {
            "affected_towers_count": len(affected_towers),
            "tower_states": [],
            "network_metrics": {
                "total_bandwidth_reallocated": sum(r.get("bandwidth_amount", 0) for r in reallocations),
                "average_execution_time": distribution_plan.get("execution_metadata", {}).get("estimated_execution_time_minutes", 0)
            }
        }
        
        for tower_id in affected_towers:
            try:
                # Get current metrics and allocations
                current_metrics = self.db_manager.get_latest_tower_metrics(tower_id)
                current_allocations = self._get_current_allocations(tower_id)
                
                if current_metrics:
                    tower_state = {
                        "tower_id": tower_id,
                        "current_load": max(current_metrics.cpu_utilization, 
                                          current_metrics.memory_usage, 
                                          current_metrics.bandwidth_usage),
                        "total_allocated_bandwidth": sum(a.allocated_bandwidth for a in current_allocations),
                        "allocation_count": len(current_allocations),
                        "congestion_status": current_metrics.congestion_severity.value
                    }
                    post_analysis["tower_states"].append(tower_state)
                    
            except Exception as e:
                logging.error(f"Error analyzing post-execution state for {tower_id}: {e}")
        
        return post_analysis
    
    # Additional helper methods for analysis and monitoring
    
    def _analyze_reallocation_effectiveness(self, action: OptimizationAction) -> Dict[str, Any]:
        """Analyze effectiveness of a reallocation action"""
        try:
            analysis = {
                "action_id": action.id,
                "action_type": action.action_type.value,
                "executed_at": action.executed_at.isoformat(),
                "effectiveness_score": action.effectiveness_score,
                "tower_ids": action.tower_ids,
                "parameters": action.parameters
            }
            
            # Extract bandwidth information from parameters
            if isinstance(action.parameters, dict):
                analysis["bandwidth_reallocated"] = action.parameters.get("bandwidth_amount", 0)
                analysis["algorithm_used"] = action.parameters.get("algorithm", "unknown")
            
            # Calculate actual impact (simplified)
            if action.effectiveness_score:
                if action.effectiveness_score >= 70:
                    analysis["impact_level"] = "high"
                elif action.effectiveness_score >= 40:
                    analysis["impact_level"] = "medium"
                else:
                    analysis["impact_level"] = "low"
            else:
                analysis["impact_level"] = "unknown"
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing reallocation effectiveness: {e}")
            return {
                "action_id": action.id,
                "effectiveness_score": None,
                "error": str(e)
            }
    
    def _identify_reallocation_trends(self, effectiveness_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify trends in reallocation effectiveness"""
        if len(effectiveness_analysis) < 2:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Analyze effectiveness trend
        valid_scores = [a["effectiveness_score"] for a in effectiveness_analysis 
                       if a.get("effectiveness_score") is not None]
        
        if len(valid_scores) < 2:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Simple trend analysis
        recent_scores = valid_scores[-3:] if len(valid_scores) >= 3 else valid_scores
        older_scores = valid_scores[:-3] if len(valid_scores) > 3 else []
        
        if older_scores:
            recent_avg = sum(recent_scores) / len(recent_scores)
            older_avg = sum(older_scores) / len(older_scores)
            
            if recent_avg > older_avg + 10:
                trend = "improving"
            elif recent_avg < older_avg - 10:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Analyze algorithm performance
        algorithm_performance = {}
        for analysis in effectiveness_analysis:
            algorithm = analysis.get("algorithm_used", "unknown")
            score = analysis.get("effectiveness_score")
            
            if algorithm not in algorithm_performance:
                algorithm_performance[algorithm] = []
            if score is not None:
                algorithm_performance[algorithm].append(score)
        
        # Calculate average performance per algorithm
        algorithm_averages = {}
        for algorithm, scores in algorithm_performance.items():
            if scores:
                algorithm_averages[algorithm] = sum(scores) / len(scores)
        
        return {
            "trend": trend,
            "confidence": min(len(valid_scores) / 10.0, 1.0),
            "recent_average": sum(recent_scores) / len(recent_scores) if recent_scores else 0,
            "overall_average": sum(valid_scores) / len(valid_scores),
            "algorithm_performance": algorithm_averages
        }
    
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
        if len(spectrum_actions) > 1:
            time_span = (spectrum_actions[0].executed_at - spectrum_actions[-1].executed_at).total_seconds() / 3600
            action_frequency = len(spectrum_actions) / time_span if time_span > 0 else 0
        else:
            time_span = 1.0
            action_frequency = 1.0
        
        return {
            "total_actions": len(spectrum_actions),
            "completed_actions": len(completed_actions),
            "success_rate": round(success_rate, 2),
            "average_effectiveness": round(avg_effectiveness, 2),
            "actions_per_hour": round(action_frequency, 2),
            "time_span_hours": round(time_span, 2)
        }
    
    def _generate_reallocation_recommendations(self, effectiveness_analysis: List[Dict[str, Any]], 
                                             trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving reallocation effectiveness"""
        recommendations = []
        
        # Trend-based recommendations
        if trends["trend"] == "declining" and trends["confidence"] > 0.5:
            recommendations.append("Reallocation effectiveness is declining - review algorithm parameters")
        
        # Algorithm performance recommendations
        algorithm_performance = trends.get("algorithm_performance", {})
        if algorithm_performance:
            best_algorithm = max(algorithm_performance.items(), key=lambda x: x[1])
            worst_algorithm = min(algorithm_performance.items(), key=lambda x: x[1])
            
            if best_algorithm[1] - worst_algorithm[1] > 20:
                recommendations.append(f"Consider using {best_algorithm[0]} algorithm more frequently - "
                                     f"it shows {best_algorithm[1]:.1f}% effectiveness vs "
                                     f"{worst_algorithm[1]:.1f}% for {worst_algorithm[0]}")
        
        # Impact level analysis
        impact_levels = {}
        for analysis in effectiveness_analysis:
            level = analysis.get("impact_level", "unknown")
            impact_levels[level] = impact_levels.get(level, 0) + 1
        
        total_actions = len(effectiveness_analysis)
        if total_actions > 0:
            low_impact_ratio = impact_levels.get("low", 0) / total_actions
            if low_impact_ratio > 0.4:
                recommendations.append("High ratio of low-impact reallocations - "
                                     "consider adjusting thresholds or algorithms")
        
        # Bandwidth utilization recommendations
        total_bandwidth = sum(a.get("bandwidth_reallocated", 0) for a in effectiveness_analysis)
        if total_bandwidth > 0:
            avg_bandwidth = total_bandwidth / len(effectiveness_analysis)
            if avg_bandwidth < 20:
                recommendations.append("Average reallocation bandwidth is low - "
                                     "consider consolidating smaller reallocations")
            elif avg_bandwidth > 200:
                recommendations.append("Average reallocation bandwidth is high - "
                                     "consider breaking down large reallocations")
        
        return recommendations if recommendations else ["Reallocation performance appears satisfactory"]
    
    def _assess_network_impact(self, reallocation_actions: List[OptimizationAction]) -> Dict[str, Any]:
        """Assess overall network impact of reallocations"""
        if not reallocation_actions:
            return {"impact": "none", "details": "No reallocation actions to assess"}
        
        # Calculate impact metrics
        total_bandwidth_moved = 0
        affected_towers = set()
        algorithm_usage = {}
        
        for action in reallocation_actions:
            if isinstance(action.parameters, dict):
                bandwidth = action.parameters.get("bandwidth_amount", 0)
                total_bandwidth_moved += bandwidth
                
                algorithm = action.parameters.get("algorithm", "unknown")
                algorithm_usage[algorithm] = algorithm_usage.get(algorithm, 0) + 1
            
            affected_towers.update(action.tower_ids)
        
        # Calculate effectiveness distribution
        effectiveness_scores = [a.effectiveness_score for a in reallocation_actions 
                              if a.effectiveness_score is not None]
        
        if effectiveness_scores:
            avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
            high_effectiveness = len([s for s in effectiveness_scores if s >= 70])
            medium_effectiveness = len([s for s in effectiveness_scores if 40 <= s < 70])
            low_effectiveness = len([s for s in effectiveness_scores if s < 40])
        else:
            avg_effectiveness = 0
            high_effectiveness = medium_effectiveness = low_effectiveness = 0
        
        # Determine overall impact level
        if avg_effectiveness >= 70 and total_bandwidth_moved > 500:
            impact_level = "high_positive"
        elif avg_effectiveness >= 50 and total_bandwidth_moved > 200:
            impact_level = "moderate_positive"
        elif avg_effectiveness >= 30:
            impact_level = "low_positive"
        else:
            impact_level = "minimal"
        
        return {
            "impact": impact_level,
            "details": {
                "total_bandwidth_moved_mhz": total_bandwidth_moved,
                "affected_towers_count": len(affected_towers),
                "average_effectiveness": round(avg_effectiveness, 2),
                "effectiveness_distribution": {
                    "high": high_effectiveness,
                    "medium": medium_effectiveness,
                    "low": low_effectiveness
                },
                "algorithm_usage": algorithm_usage,
                "total_actions": len(reallocation_actions)
            }
        }
    
    def _calculate_tower_efficiency(self, allocations: List[SpectrumAllocation], 
                                  metrics: TowerMetrics) -> float:
        """Calculate efficiency score for a tower's spectrum allocations"""
        if not allocations:
            return 0.0
        
        # Calculate utilization efficiency
        total_allocated = sum(a.allocated_bandwidth for a in allocations)
        avg_utilization = sum(a.utilization_percentage for a in allocations) / len(allocations)
        
        # Efficiency based on how well allocated spectrum matches actual usage
        usage_efficiency = min(metrics.bandwidth_usage / 80.0, 1.0)  # 80% is optimal
        allocation_efficiency = avg_utilization / 100.0
        
        # Signal quality factor
        signal_factor = max(0, (metrics.signal_strength + 100) / 30)  # -100 to -70 dBm range
        
        # Error rate penalty
        error_penalty = max(0, 1 - (metrics.error_rate / 10))
        
        # Combined efficiency score
        efficiency = (usage_efficiency * 0.4 + allocation_efficiency * 0.4 + 
                     signal_factor * 0.1 + error_penalty * 0.1) * 100
        
        return min(100.0, max(0.0, efficiency))
    
    def _analyze_allocation_efficiency(self, allocations: List[SpectrumAllocation], 
                                     metrics: TowerMetrics) -> List[Dict[str, Any]]:
        """Analyze allocation efficiency and suggest optimizations"""
        optimizations = []
        
        if not allocations:
            return optimizations
        
        # Check for band optimization opportunities
        band_utilization = {}
        for allocation in allocations:
            band = allocation.frequency_band
            if band not in band_utilization:
                band_utilization[band] = {"total_bandwidth": 0, "avg_utilization": 0, "count": 0}
            
            band_utilization[band]["total_bandwidth"] += allocation.allocated_bandwidth
            band_utilization[band]["avg_utilization"] += allocation.utilization_percentage
            band_utilization[band]["count"] += 1
        
        # Calculate averages
        for band_info in band_utilization.values():
            if band_info["count"] > 0:
                band_info["avg_utilization"] /= band_info["count"]
        
        # Suggest band reallocations
        for band, info in band_utilization.items():
            if info["avg_utilization"] < 30 and info["total_bandwidth"] > 50:
                # Low utilization, consider moving to more efficient band
                target_band = self._suggest_optimal_band(metrics, info["total_bandwidth"])
                if target_band != band:
                    optimizations.append({
                        "type": "reallocate_band",
                        "from_band": band,
                        "to_band": target_band,
                        "bandwidth": info["total_bandwidth"],
                        "efficiency_gain": 15  # Estimated gain
                    })
        
        # Suggest consolidation if many small allocations
        if len(allocations) > 3:
            small_allocations = [a for a in allocations if a.allocated_bandwidth < 30]
            if len(small_allocations) >= 2:
                total_small_bandwidth = sum(a.allocated_bandwidth for a in small_allocations)
                optimizations.append({
                    "type": "consolidate_allocations",
                    "bands": [a.frequency_band for a in small_allocations],
                    "target_band": "mid_band",  # Default consolidation target
                    "total_bandwidth": total_small_bandwidth,
                    "efficiency_gain": 10  # Estimated gain
                })
        
        return optimizations
    
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
    
    def _suggest_optimal_band(self, metrics: TowerMetrics, bandwidth_needed: float) -> str:
        """Suggest optimal frequency band based on metrics and bandwidth needs"""
        # Simple heuristic based on signal strength and bandwidth requirements
        if metrics.signal_strength > -75 and bandwidth_needed > 150:
            return "high_band"  # Good signal, high bandwidth need
        elif bandwidth_needed > 50:
            return "mid_band"   # Balanced option
        else:
            return "low_band"   # Wide coverage for low bandwidth
    
    def _analyze_historical_patterns(self, tower_id: str) -> Dict[str, Any]:
        """Analyze historical patterns for predictive allocation (simplified)"""
        try:
            # Get recent metrics history
            metrics_history = self.db_manager.get_tower_metrics_history(tower_id, hours=48)
            
            if len(metrics_history) < 5:
                return {"trend": "insufficient_data", "confidence": 0.0}
            
            # Calculate load trend
            loads = [max(m.cpu_utilization, m.memory_usage, m.bandwidth_usage) for m in metrics_history]
            
            # Simple trend calculation
            recent_loads = loads[:len(loads)//2]  # More recent half
            older_loads = loads[len(loads)//2:]   # Older half
            
            recent_avg = sum(recent_loads) / len(recent_loads)
            older_avg = sum(older_loads) / len(older_loads)
            
            load_change = recent_avg - older_avg
            
            if load_change > 5:
                trend = "increasing"
                predicted_increase = load_change * 1.5  # Extrapolate
            elif load_change < -5:
                trend = "decreasing"
                predicted_increase = load_change * 1.5
            else:
                trend = "stable"
                predicted_increase = 0
            
            confidence = min(len(metrics_history) / 20.0, 1.0)  # More data = higher confidence
            
            return {
                "trend": trend,
                "confidence": confidence,
                "predicted_load_increase": predicted_increase,
                "current_average_load": recent_avg
            }
            
        except Exception as e:
            logging.error(f"Error analyzing historical patterns for {tower_id}: {e}")
            return {"trend": "error", "confidence": 0.0}
    
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