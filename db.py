import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect("detections.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS person_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        time TEXT,
        person_count INTEGER
    )
""")
conn.commit()

# Save a detection record
def save_to_db(person_count):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_12hr = now.strftime("%I:%M:%S %p")

    cursor.execute(
        "INSERT INTO person_detections (date, time, person_count) VALUES (?, ?, ?)",
        (date, time_12hr, person_count)
    )
    conn.commit()
