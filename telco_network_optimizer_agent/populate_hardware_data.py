"""
Populate sample hardware status data for testing the Maintenance MCP Server
"""

import sqlite3
import os
from datetime import datetime, timedelta
import random
import json

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")

def populate_hardware_status_data():
    """Populate the database with sample hardware status data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        print("Populating sample hardware status data...")
        
        # Get existing towers
        cursor.execute("SELECT id FROM towers")
        tower_ids = [row[0] for row in cursor.fetchall()]
        
        if not tower_ids:
            print("No towers found. Please run the schema migration first.")
            return
        
        # Component types and their typical status distributions
        components = {
            "ANTENNA": {"HEALTHY": 0.7, "WARNING": 0.2, "CRITICAL": 0.08, "FAILED": 0.02},
            "PROCESSOR": {"HEALTHY": 0.6, "WARNING": 0.25, "CRITICAL": 0.12, "FAILED": 0.03},
            "MEMORY": {"HEALTHY": 0.75, "WARNING": 0.15, "CRITICAL": 0.08, "FAILED": 0.02},
            "POWER": {"HEALTHY": 0.8, "WARNING": 0.15, "CRITICAL": 0.04, "FAILED": 0.01},
            "COOLING": {"HEALTHY": 0.65, "WARNING": 0.25, "CRITICAL": 0.08, "FAILED": 0.02},
            "NETWORK": {"HEALTHY": 0.7, "WARNING": 0.2, "CRITICAL": 0.08, "FAILED": 0.02}
        }
        
        # Clear existing hardware status data
        cursor.execute("DELETE FROM hardware_status")
        
        current_time = datetime.now()
        hardware_status_data = []
        
        for tower_id in tower_ids:
            for component_type, status_dist in components.items():
                # Generate multiple status entries over time (last 24 hours)
                for i in range(5):  # 5 entries per component per tower
                    timestamp = current_time - timedelta(hours=i*4 + random.randint(0, 60)/60)
                    
                    # Select status based on distribution
                    rand_val = random.random()
                    cumulative = 0
                    selected_status = "HEALTHY"
                    
                    for status, prob in status_dist.items():
                        cumulative += prob
                        if rand_val <= cumulative:
                            selected_status = status
                            break
                    
                    # Generate appropriate error codes and metrics
                    error_codes = None
                    performance_metrics = {}
                    temperature = None
                    
                    if selected_status == "FAILED":
                        error_codes = f"CRIT_{random.randint(100, 999)}"
                    elif selected_status == "CRITICAL":
                        error_codes = f"ERR_{random.randint(100, 999)}"
                    elif selected_status == "WARNING":
                        error_codes = f"WARN_{random.randint(100, 999)}"
                    
                    # Component-specific metrics
                    if component_type == "ANTENNA":
                        signal_strength = -65 if selected_status == "HEALTHY" else random.randint(-90, -70)
                        performance_metrics["signal_strength"] = signal_strength
                        performance_metrics["vswr"] = round(random.uniform(1.1, 2.5), 2)
                        
                    elif component_type == "PROCESSOR":
                        cpu_usage = random.randint(30, 60) if selected_status == "HEALTHY" else random.randint(70, 95)
                        performance_metrics["cpu_usage"] = cpu_usage
                        performance_metrics["load_average"] = round(random.uniform(0.5, 3.0), 2)
                        temperature = random.uniform(45, 85)
                        
                    elif component_type == "MEMORY":
                        memory_usage = random.randint(40, 70) if selected_status == "HEALTHY" else random.randint(80, 95)
                        performance_metrics["memory_usage"] = memory_usage
                        performance_metrics["swap_usage"] = random.randint(0, 30)
                        
                    elif component_type == "POWER":
                        voltage = round(random.uniform(11.8, 12.2), 1) if selected_status == "HEALTHY" else round(random.uniform(10.0, 11.5), 1)
                        performance_metrics["voltage"] = voltage
                        performance_metrics["current"] = round(random.uniform(8.0, 15.0), 1)
                        performance_metrics["power_consumption"] = round(voltage * performance_metrics["current"], 1)
                        
                    elif component_type == "COOLING":
                        temp = random.uniform(25, 45) if selected_status == "HEALTHY" else random.uniform(50, 75)
                        temperature = temp
                        performance_metrics["fan_speed"] = random.randint(1200, 3000)
                        performance_metrics["airflow"] = round(random.uniform(50, 150), 1)
                        
                    elif component_type == "NETWORK":
                        latency = random.uniform(1, 10) if selected_status == "HEALTHY" else random.uniform(15, 50)
                        performance_metrics["latency_ms"] = round(latency, 2)
                        performance_metrics["packet_loss"] = round(random.uniform(0, 0.1) if selected_status == "HEALTHY" else random.uniform(0.5, 5.0), 3)
                        performance_metrics["throughput_mbps"] = random.randint(800, 1000) if selected_status == "HEALTHY" else random.randint(200, 700)
                    
                    hardware_status_data.append((
                        tower_id,
                        component_type,
                        selected_status,
                        timestamp.isoformat(),
                        error_codes,
                        json.dumps(performance_metrics) if performance_metrics else None,
                        temperature
                    ))
        
        # Insert all hardware status data
        cursor.executemany("""
            INSERT INTO hardware_status 
            (tower_id, component_type, status, last_checked, error_codes, performance_metrics, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, hardware_status_data)
        
        conn.commit()
        print(f"✅ Inserted {len(hardware_status_data)} hardware status records")
        
        # Display summary
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM hardware_status 
            GROUP BY status 
            ORDER BY count DESC
        """)
        status_summary = cursor.fetchall()
        
        print("\nHardware Status Summary:")
        for status, count in status_summary:
            print(f"  {status}: {count} records")
        
        # Display tower-wise summary
        cursor.execute("""
            SELECT tower_id, COUNT(*) as total_components,
                   SUM(CASE WHEN status = 'HEALTHY' THEN 1 ELSE 0 END) as healthy,
                   SUM(CASE WHEN status = 'WARNING' THEN 1 ELSE 0 END) as warning,
                   SUM(CASE WHEN status = 'CRITICAL' THEN 1 ELSE 0 END) as critical,
                   SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed
            FROM hardware_status 
            GROUP BY tower_id 
            ORDER BY tower_id
        """)
        tower_summary = cursor.fetchall()
        
        print("\nTower-wise Hardware Status:")
        for row in tower_summary:
            tower_id, total, healthy, warning, critical, failed = row
            print(f"  {tower_id}: {total} records - H:{healthy} W:{warning} C:{critical} F:{failed}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error populating hardware status data: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    populate_hardware_status_data()