import sqlite3
import hashlib

conn = sqlite3.connect("reports.db")
cursor = conn.cursor()

# Get all users
cursor.execute("SELECT id, password FROM users")
users = cursor.fetchall()

for user_id, password in users:
    # Skip if already hashed (length of sha256 = 64)
    if len(password) == 64:
        continue

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    cursor.execute(
        "UPDATE users SET password=? WHERE id=?",
        (hashed_password, user_id)
    )

conn.commit()
conn.close()

print("✅ Password migration completed successfully!")