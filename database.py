import sqlite3

def create_table():
    conn = sqlite3.connect("reports.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        prediction TEXT,
        confidence REAL,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()


def insert_report(patient_name, prediction, confidence, date):
    conn = sqlite3.connect("reports.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO reports (patient_name, prediction, confidence, date)
    VALUES (?, ?, ?, ?)
    """, (patient_name, prediction, confidence, date))

    conn.commit()
    conn.close()


def get_reports():
    conn = sqlite3.connect("reports.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM reports ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data