"""
db_manager.py

users
├── id (PK)
└── name

embeddings                 # несколько embeddings на одного пользователя
├── id (PK)
├── user_id (FK -> users.id)
└── vector (BLOB)           # numpy array, float32, сериализован через .tobytes()

attendance
├── id (PK)
├── user_id (FK -> users.id)
├── date
└── time
"""

import sqlite3
import numpy as np
import os
from datetime import datetime

import config


def get_connection():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            vector BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()


# ---------- users / embeddings ----------

def add_user(name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id


def add_embedding(user_id, embedding_vector):
    """embedding_vector: numpy array формы (config.EMBEDDING_SIZE,)"""
    conn = get_connection()
    cursor = conn.cursor()
    vector_blob = np.asarray(embedding_vector, dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO embeddings (user_id, vector) VALUES (?, ?)",
        (user_id, vector_blob)
    )
    conn.commit()
    conn.close()


def get_all_embeddings():
    """Возвращает [(user_id, name, embedding_vector), ...] для всех пользователей."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT embeddings.user_id, users.name, embeddings.vector
        FROM embeddings
        JOIN users ON embeddings.user_id = users.id
    """)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for user_id, name, vector_blob in rows:
        vector = np.frombuffer(vector_blob, dtype=np.float32)
        results.append((user_id, name, vector))
    return results


# ---------- attendance ----------

def mark_attendance(user_id):
    """
    Записывает посещение с текущей датой/временем.
    Возвращает False, если пользователь уже отмечен сегодня (защита от дублей).
    """
    conn = get_connection()
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    cursor.execute(
        "SELECT id FROM attendance WHERE user_id = ? AND date = ?",
        (user_id, today)
    )
    if cursor.fetchone():
        conn.close()
        return False

    cursor.execute(
        "INSERT INTO attendance (user_id, date, time) VALUES (?, ?, ?)",
        (user_id, today, current_time)
    )
    conn.commit()
    conn.close()
    return True


def get_today_attendance():
    """Возвращает [(name, time), ...] за сегодня, отсортировано по времени."""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT users.name, attendance.time
        FROM attendance
        JOIN users ON attendance.user_id = users.id
        WHERE attendance.date = ?
        ORDER BY attendance.time
    """, (today,))
    rows = cursor.fetchall()
    conn.close()
    return rows
