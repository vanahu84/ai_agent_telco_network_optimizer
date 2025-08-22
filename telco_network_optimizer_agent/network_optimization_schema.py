"""
Network Optimization Database Schema Migration
Extends the existing telecom.db with network optimization tables
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")

def migrate_database():
    """
    Migrate the existing telecom database to include network optimization tables
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        print("Starting network optimization schema migration...")
        
        # 1. Extend existing customers table with location and service tier
        print("Extending customers table...")
        try:
            cursor.execute("ALTER TABLE customers ADD COLUMN location_area TEXT")
            cursor.execute("ALTER TABLE customers ADD COLUMN service_tier TEXT DEFAULT 'standard'")
            print("✓ Extended customers table with location_area and service_tier")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("✓ Customers table already extended")
            else:
                raise
        
        # 2. Create towers table
        print("Creating towers table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS towers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                coverage_radius REAL DEFAULT 2.0,
                max_capacity INTEGER DEFAULT 1000,
                technology TEXT DEFAULT '5G',
                status TEXT DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'MAINTENANCE', 'OFFLINE'))
            )
        """)
        print("✓ Created towers table")
        
        # 3. Create tower_metrics table for real-time monitoring
        print("Creating tower_metrics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tower_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                cpu_utilization REAL CHECK(cpu_utilization >= 0 AND cpu_utilization <= 100),
                memory_usage REAL CHECK(memory_usage >= 0 AND memory_usage <= 100),
                bandwidth_usage REAL CHECK(bandwidth_usage >= 0 AND bandwidth_usage <= 100),
                active_connections INTEGER DEFAULT 0,
                signal_strength REAL DEFAULT -70.0,
                error_rate REAL DEFAULT 0.0 CHECK(error_rate >= 0 AND error_rate <= 100),
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        print("✓ Created tower_metrics table")
        
        # 4. Create congestion_events table
        print("Creating congestion_events table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS congestion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                severity TEXT NOT NULL CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH')),
                detected_at DATETIME NOT NULL,
                resolved_at DATETIME,
                metrics_snapshot TEXT,
                affected_area TEXT,
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        print("✓ Created congestion_events table")
        
        # 5. Create spectrum_allocations table
        print("Creating spectrum_allocations table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spectrum_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                frequency_band TEXT NOT NULL,
                allocated_bandwidth REAL NOT NULL CHECK(allocated_bandwidth > 0),
                utilization_percentage REAL DEFAULT 0.0 CHECK(utilization_percentage >= 0 AND utilization_percentage <= 100),
                allocation_timestamp DATETIME NOT NULL,
                expires_at DATETIME,
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        print("✓ Created spectrum_allocations table")
        
        # 6. Create optimization_actions table
        print("Creating optimization_actions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL CHECK(action_type IN ('SPECTRUM_REALLOCATION', 'LOAD_BALANCING', 'TRAFFIC_REDIRECT', 'MAINTENANCE_TRIGGER')),
                tower_ids TEXT NOT NULL,
                parameters TEXT,
                executed_at DATETIME NOT NULL,
                effectiveness_score REAL CHECK(effectiveness_score >= 0 AND effectiveness_score <= 100),
                status TEXT DEFAULT 'PENDING' CHECK(status IN ('PENDING', 'EXECUTING', 'COMPLETED', 'FAILED'))
            )
        """)
        print("✓ Created optimization_actions table")
        
        # 7. Create user_movement_patterns table
        print("Creating user_movement_patterns table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_movement_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                area_id TEXT NOT NULL,
                hour_of_day INTEGER CHECK(hour_of_day >= 0 AND hour_of_day <= 23),
                day_of_week INTEGER CHECK(day_of_week >= 0 AND day_of_week <= 6),
                average_users INTEGER DEFAULT 0,
                peak_users INTEGER DEFAULT 0,
                pattern_confidence REAL DEFAULT 0.0 CHECK(pattern_confidence >= 0 AND pattern_confidence <= 1.0),
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✓ Created user_movement_patterns table")
        
        # 8. Create traffic_predictions table
        print("Creating traffic_predictions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                prediction_timestamp DATETIME NOT NULL,
                predicted_load REAL CHECK(predicted_load >= 0 AND predicted_load <= 100),
                confidence_level REAL CHECK(confidence_level >= 0 AND confidence_level <= 1.0),
                prediction_horizon_minutes INTEGER DEFAULT 60,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        print("✓ Created traffic_predictions table")
        
        # 9. Create hardware_status table
        print("Creating hardware_status table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                component_type TEXT NOT NULL CHECK(component_type IN ('ANTENNA', 'PROCESSOR', 'MEMORY', 'POWER', 'COOLING', 'NETWORK')),
                status TEXT NOT NULL CHECK(status IN ('HEALTHY', 'WARNING', 'CRITICAL', 'FAILED')),
                last_checked DATETIME NOT NULL,
                error_codes TEXT,
                performance_metrics TEXT,
                temperature REAL,
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        print("✓ Created hardware_status table")
        
        # 10. Create maintenance_tickets table
        print("Creating maintenance_tickets table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                priority TEXT NOT NULL CHECK(priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                issue_description TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                assigned_to TEXT,
                status TEXT DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'ASSIGNED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED')),
                resolved_at DATETIME,
                resolution_notes TEXT,
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        print("✓ Created maintenance_tickets table")
        
        # 11. Create network_events table for general event logging
        print("Creating network_events table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS network_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL CHECK(event_type IN ('CONGESTION', 'OPTIMIZATION', 'MAINTENANCE', 'ALERT', 'SYSTEM')),
                tower_id TEXT,
                timestamp DATETIME NOT NULL,
                severity TEXT CHECK(severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
                description TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                metadata TEXT,
                FOREIGN KEY (tower_id) REFERENCES towers(id)
            )
        """)
        print("✓ Created network_events table")
        
        # Create indexes for better performance
        print("Creating database indexes...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tower_metrics_tower_timestamp ON tower_metrics(tower_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_congestion_events_tower_detected ON congestion_events(tower_id, detected_at)",
            "CREATE INDEX IF NOT EXISTS idx_spectrum_allocations_tower ON spectrum_allocations(tower_id)",
            "CREATE INDEX IF NOT EXISTS idx_traffic_predictions_tower_timestamp ON traffic_predictions(tower_id, prediction_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_hardware_status_tower_checked ON hardware_status(tower_id, last_checked)",
            "CREATE INDEX IF NOT EXISTS idx_maintenance_tickets_tower_status ON maintenance_tickets(tower_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_network_events_timestamp ON network_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_network_events_tower_type ON network_events(tower_id, event_type)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        print("✓ Created performance indexes")
        
        conn.commit()
        print("✅ Network optimization schema migration completed successfully!")
        
        # Display table count
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        print(f"Database now contains {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
            
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        conn.close()

def populate_sample_data():
    """
    Populate the database with sample network optimization data
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        print("Populating sample network optimization data...")
        
        # Sample towers
        sample_towers = [
            ("TOWER_001", "Downtown Central", 40.7128, -74.0060, 2.5, 1500, "5G", "ACTIVE"),
            ("TOWER_002", "Business District", 40.7589, -73.9851, 2.0, 1200, "5G", "ACTIVE"),
            ("TOWER_003", "Residential North", 40.7831, -73.9712, 3.0, 1000, "5G", "ACTIVE"),
            ("TOWER_004", "Airport Hub", 40.6892, -74.1745, 4.0, 2000, "5G", "ACTIVE"),
            ("TOWER_005", "University Campus", 40.8176, -73.9782, 2.2, 1300, "5G", "MAINTENANCE")
        ]
        
        cursor.executemany("""
            INSERT OR IGNORE INTO towers (id, name, latitude, longitude, coverage_radius, max_capacity, technology, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_towers)
        
        # Sample tower metrics (recent data)
        from datetime import timedelta
        current_time = datetime.now()
        sample_metrics = []
        for tower_id in ["TOWER_001", "TOWER_002", "TOWER_003", "TOWER_004"]:
            for i in range(5):  # 5 recent measurements per tower
                timestamp = current_time - timedelta(minutes=i*10)
                cpu_util = 45 + (i * 5) + (hash(tower_id) % 20)
                memory_usage = 60 + (i * 3) + (hash(tower_id) % 15)
                bandwidth_usage = 70 + (i * 4) + (hash(tower_id) % 25)
                
                sample_metrics.append((
                    tower_id, timestamp.isoformat(),
                    min(cpu_util, 95), min(memory_usage, 90), min(bandwidth_usage, 95),
                    800 + (i * 50), -65.0 - (i * 2), 0.1 + (i * 0.05)
                ))
        
        cursor.executemany("""
            INSERT INTO tower_metrics (tower_id, timestamp, cpu_utilization, memory_usage, bandwidth_usage, active_connections, signal_strength, error_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_metrics)
        
        # Sample congestion events
        sample_congestion = [
            ("TOWER_001", "HIGH", (current_time - timedelta(hours=2)).isoformat(), None, '{"cpu": 95, "bandwidth": 92}', "Downtown Central"),
            ("TOWER_002", "MEDIUM", (current_time - timedelta(hours=1)).isoformat(), (current_time - timedelta(minutes=30)).isoformat(), '{"cpu": 87, "bandwidth": 85}', "Business District")
        ]
        
        cursor.executemany("""
            INSERT INTO congestion_events (tower_id, severity, detected_at, resolved_at, metrics_snapshot, affected_area)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sample_congestion)
        
        # Update some customers with location areas
        cursor.execute("UPDATE customers SET location_area = 'downtown', service_tier = 'premium' WHERE id <= 3")
        cursor.execute("UPDATE customers SET location_area = 'business', service_tier = 'standard' WHERE id > 3 AND id <= 6")
        cursor.execute("UPDATE customers SET location_area = 'residential', service_tier = 'basic' WHERE id > 6")
        
        conn.commit()
        print("✅ Sample data populated successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Sample data population failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
    populate_sample_data()