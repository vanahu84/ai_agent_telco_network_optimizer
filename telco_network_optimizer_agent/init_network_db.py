#!/usr/bin/env python3
"""
Network Optimization Database Initialization Script
Run this script to set up the complete network optimization database schema
"""

import os
import sys
from datetime import datetime

def main():
    """Initialize the network optimization database"""
    print("🚀 Initializing Network Optimization Database...")
    print("=" * 60)
    
    try:
        # Import and run migration
        from network_optimization_schema import migrate_database, populate_sample_data
        
        print("📊 Running database migration...")
        migrate_database()
        
        print("\n📝 Populating sample data...")
        populate_sample_data()
        
        print("\n🧪 Testing database operations...")
        from network_db_utils import get_db_manager
        
        db_manager = get_db_manager()
        
        # Test basic operations
        towers = db_manager.get_all_towers()
        print(f"✅ Found {len(towers)} towers in database")
        
        # Test metrics
        if towers:
            tower_id = towers[0].id
            metrics = db_manager.get_latest_tower_metrics(tower_id)
            if metrics:
                print(f"✅ Latest metrics for {tower_id}: CPU={metrics.cpu_utilization:.1f}%, Bandwidth={metrics.bandwidth_usage:.1f}%")
        
        # Test network health
        health = db_manager.get_network_health_overview()
        print(f"✅ Network health score: {health['network_health_score']:.1f}/100")
        
        print("\n" + "=" * 60)
        print("🎉 Network Optimization Database initialized successfully!")
        print("\nDatabase contains the following tables:")
        
        import sqlite3
        DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        for table in tables:
            if table[0] != 'sqlite_sequence':
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"  📋 {table[0]}: {count} records")
        
        conn.close()
        
        print("\n💡 You can now use the network optimization MCP servers!")
        print("   - Run MCP servers to start network monitoring")
        print("   - Use network_db_utils.py for database operations")
        print("   - Check network_models.py for data structures")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()