import sqlite3

conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM person_detections")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
