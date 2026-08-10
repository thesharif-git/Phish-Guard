from data.database import engine
from sqlmodel import text

with engine.connect() as conn:
    # Fix: sharifhassan4266@gmail.com is actually a GitHub user
    conn.execute(text("UPDATE user SET auth_provider = 'github' WHERE email = 'sharifhassan4266@gmail.com'"))
    # Also fix meetmuhammadsharif@gmail.com — clear it (unknown provider)
    conn.execute(text("UPDATE user SET auth_provider = NULL WHERE email = 'meetmuhammadsharif@gmail.com'"))
    conn.commit()

    result = conn.execute(text("SELECT email, auth_provider FROM user"))
    for row in result:
        print(f"Email: {row[0]}, Provider: {row[1]}")
    print("Fix complete.")
