import sqlite3
from typing import Optional
from pydantic import BaseModel

DB_NAME = "users.db"

class User(BaseModel):
    username: str
    password: str
    full_name: str

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, full_name TEXT)''')
    conn.commit()
    conn.close()

def create_user(user: User) -> bool:
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)",
                  (user.username, user.password, user.full_name))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user(username: str) -> Optional[User]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, password, full_name FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return User(username=row[0], password=row[1], full_name=row[2])
    return None
