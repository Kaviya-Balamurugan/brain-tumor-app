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

def create_user_table():
    conn = sqlite3.connect("reports.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


def add_user(username, password):
    conn = sqlite3.connect("reports.db")
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    conn = sqlite3.connect("reports.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()
    conn.close()

    return user