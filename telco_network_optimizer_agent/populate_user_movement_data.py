#!/usr/bin/env python3
"""
Populate sample user movement data for testing
"""

import sqlite3
import os
from datetime import datetime, timedelta

DATABASE_PATH = os.path.join(os.path.dirname(__file__), "telecom.db")

def populate_user_movement_data():
    """Populate sample user movement patterns"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Clear existing data
        cursor.execute("DELETE FROM user_movement_patterns")
        
        # Insert sample movement patterns
        current_time = datetime.now()
        areas = ["downtown", "business", "residential", "airport", "university"]
        
        for area in areas:
            for hour in range(24):
                for day in range(7):
                    # Simulate realistic patterns
                    base_users = 50
                    if area == "downtown":
                        base_users = 100
                    elif area == "business":
                        base_users = 80
                    elif area == "airport":
                        base_users = 120
                    
                    # Add time-based variations
                    if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                        multiplier = 1.5
                    elif 22 <= hour or hour <= 6:  # Night hours
                        multiplier = 0.3
                    else:
                        multiplier = 1.0
                    
                    # Weekend variations
                    if day >= 5:  # Weekend
                        if area in ["business"]:
                            multiplier *= 0.4
                        elif area in ["downtown", "residential"]:
                            multiplier *= 1.2
                    
                    avg_users = int(base_users * multiplier)
                    peak_users = int(avg_users * 1.4)
                    confidence = 0.7 + (hour % 3) * 0.1
                    
                    cursor.execute("""
                        INSERT INTO user_movement_patterns 
                        (area_id, hour_of_day, day_of_week, average_users, peak_users, pattern_confidence, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (area, hour, day, avg_users, peak_users, confidence, current_time.isoformat()))
        
        conn.commit()
        print(f"✅ Populated user movement patterns for {len(areas)} areas")
        
        # Verify data
        cursor.execute("SELECT COUNT(*) FROM user_movement_patterns")
        count = cursor.fetchone()[0]
        print(f"✅ Total patterns inserted: {count}")
        
    except Exception as e:
        print(f"❌ Error populating data: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    populate_user_movement_data()