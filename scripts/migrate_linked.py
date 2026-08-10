from data.database import engine
from sqlmodel import text

with engine.connect() as conn:
    # Backfill from auth_provider field
    conn.execute(text("UPDATE user SET google_linked = 1 WHERE auth_provider = 'google'"))
    conn.execute(text("UPDATE user SET github_linked = 1 WHERE auth_provider = 'github'"))

    # YOUR account: sharifhassan4266@gmail.com — logs in via GitHub
    conn.execute(text("UPDATE user SET github_linked = 1, auth_provider = 'github' WHERE email = 'sharifhassan4266@gmail.com'"))
    conn.execute(text("UPDATE user SET google_linked = 0 WHERE email = 'sharifhassan4266@gmail.com'"))

    conn.commit()

    result = conn.execute(text("SELECT email, auth_provider, google_linked, github_linked FROM user"))
    for row in result:
        print(f"email={row[0]}, provider={row[1]}, google={row[2]}, github={row[3]}")
    print("Done.")
