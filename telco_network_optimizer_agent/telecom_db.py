import sqlite3
import os
from datetime import datetime, timedelta
import random

# Define the database file path
DATABASE_NAME = "telecom.db"
DATABASE_PATH = os.path.join(os.path.dirname(__file__), DATABASE_NAME)

# Remove existing DB for a fresh start (optional)
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)
    print(f"Old database '{DATABASE_NAME}' deleted.")

# Connect to the new database
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# Create customers table
cursor.execute("""
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT
    )
""")

# Create orders table
cursor.execute("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        order_type TEXT CHECK(order_type IN ('Create Account', 'Sim Swap', 'Package Conv')),
        order_status TEXT CHECK(order_status IN ('Applied', 'Pending', 'Open', 'Completed')),
        created_at TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    )
""")

# Dummy customer data
customers = [
    ("John Doe", "9876543210", "john@example.com"),
    ("Jane Smith", "9123456789", "jane@example.com"),
    ("Ali Khan", "9988776655", "ali@example.com"),
    ("Maria Garcia", "9090909090", "maria@example.com"),
    ("Wei Zhang", "8008008000", "wei@example.com"),
    ("Emily Davis", "7007007007", "emily@example.com"),
    ("Ahmed Ibrahim", "9998887776", "ahmed@example.com"),
    ("Sara Lee", "9887766554", "sara@example.com"),
    ("Carlos Ramirez", "9665554433", "carlos@example.com"),
    ("Nina Patel", "9554433221", "nina@example.com")
]

# Insert customers
cursor.executemany("INSERT INTO customers (name, phone, email) VALUES (?, ?, ?)", customers)

# Possible values
order_types = ["Create Account", "Sim Swap", "Package Conv"]
order_statuses = ["Applied", "Pending", "Open", "Completed"]

# Generate dummy orders
orders = []
base_date = datetime(2025, 6, 1)
for i in range(15):
    customer_id = random.randint(1, len(customers))
    order_type = random.choice(order_types)
    order_status = random.choice(order_statuses)
    created_at = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
    orders.append((customer_id, order_type, order_status, created_at))

# Insert orders
cursor.executemany(
    "INSERT INTO orders (customer_id, order_type, order_status, created_at) VALUES (?, ?, ?, ?)",
    orders
)

conn.commit()
conn.close()
print(f"Database '{DATABASE_NAME}' created with customers and orders.")
 