"""
Debug script for tower load monitoring
"""

from test_tower_load_simple import TowerLoadMonitorSimple
import tempfile
import os
import sqlite3
from datetime import datetime
from unittest.mock import patch

def debug_tower_monitoring():
    # Create test database
    test_db_fd, test_db_path = tempfile.mkstemp(suffix='.db')
    
    try:
        # Set up database
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE towers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                coverage_radius REAL DEFAULT 2.0,
                max_capacity INTEGER DEFAULT 1000,
                technology TEXT DEFAULT '5G',
                status TEXT DEFAULT 'ACTIVE'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE tower_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tower_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                cpu_utilization REAL,
                memory_usage REAL,
                bandwidth_usage REAL,
                active_connections INTEGER DEFAULT 0,
                signal_strength REAL DEFAULT -70.0,
                error_rate REAL DEFAULT 0.0
            )
        """)
        
        # Insert test data
        current_time = datetime.now()
        cursor.execute("""
            INSERT INTO towers (id, name, latitude, longitude, status)
            VALUES ('TEST_TOWER_001', 'Test Tower 1', 40.7128, -74.0060, 'ACTIVE')
        """)
        
        cursor.execute("""
            INSERT INTO tower_metrics (tower_id, timestamp, cpu_utilization, memory_usage, bandwidth_usage, active_connections, signal_strength, error_rate)
            VALUES ('TEST_TOWER_001', ?, 85.0, 75.0, 90.0, 800, -65.0, 0.1)
        """, (current_time.isoformat(),))
        
        conn.commit()
        conn.close()
        
        # Test with patched database path
        with patch('network_db_utils.DATABASE_PATH', test_db_path):
            monitor = TowerLoadMonitorSimple()
            result = monitor.monitor_tower_load('TEST_TOWER_001')
            print('Monitor result:', result)
            
            # Test congestion detection
            congestion_result = monitor.detect_congestion(80.0)
            print('Congestion result:', congestion_result)
    
    finally:
        # Clean up
        os.close(test_db_fd)
        os.unlink(test_db_path)

if __name__ == "__main__":
    debug_tower_monitoring()