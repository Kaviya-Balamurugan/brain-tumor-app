import sqlite3
import hashlib

DB_NAME = "reports.db"

# ================= REPORT TABLE =================
def create_table():
    with sqlite3.connect(DB_NAME) as conn:
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


def insert_report(patient_name, prediction, confidence, date):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO reports (patient_name, prediction, confidence, date)
        VALUES (?, ?, ?, ?)
        """, (patient_name, prediction, confidence, date))


def get_reports():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM reports ORDER BY id DESC")
        data = cursor.fetchall()

    return data


# ================= USER TABLE =================
def create_user_table():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
        """)


def add_user(username, password):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        # 🔐 Hash password (NO change to function signature)
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed_password)
            )
            return True
        except:
            return False


def verify_user(username, password):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        # 🔐 Hash password before checking
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, hashed_password)
        )

        user = cursor.fetchone()

    return user