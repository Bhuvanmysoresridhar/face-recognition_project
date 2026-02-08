"""
SQLite database backend for structured detection history and face management.
"""

import sqlite3
import os
from datetime import datetime


class Database:
    """SQLite database for face recognition data."""

    def __init__(self, db_path="data/face_recognition.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                image_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                name TEXT NOT NULL,
                confidence REAL NOT NULL,
                distance REAL,
                camera_index INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                name TEXT NOT NULL,
                check_in TEXT NOT NULL,
                check_out TEXT,
                date TEXT NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            );

            CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
            CREATE INDEX IF NOT EXISTS idx_detections_name ON detections(name);
            CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
            CREATE INDEX IF NOT EXISTS idx_attendance_name ON attendance(name);
        """)
        self.conn.commit()

    # ---- Person management ----

    def add_person(self, name):
        try:
            self.conn.execute(
                "INSERT INTO persons (name, created_at) VALUES (?, ?)",
                (name, datetime.now().isoformat()),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # already exists

    def get_person(self, name):
        row = self.conn.execute(
            "SELECT * FROM persons WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_persons(self):
        rows = self.conn.execute("SELECT * FROM persons ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def remove_person(self, name):
        self.conn.execute("DELETE FROM persons WHERE name = ?", (name,))
        self.conn.commit()

    def update_image_count(self, name, count):
        self.conn.execute(
            "UPDATE persons SET image_count = ? WHERE name = ?", (count, name)
        )
        self.conn.commit()

    # ---- Detection logging ----

    def log_detection(self, name, confidence, distance=None, camera_index=0):
        person = self.get_person(name)
        person_id = person["id"] if person else None
        self.conn.execute(
            "INSERT INTO detections (person_id, name, confidence, distance, camera_index, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (person_id, name, confidence, distance, camera_index, datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_detections(self, name=None, start_date=None, end_date=None, limit=100):
        query = "SELECT * FROM detections WHERE 1=1"
        params = []
        if name:
            query += " AND name = ?"
            params.append(name)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_detection_stats(self):
        rows = self.conn.execute("""
            SELECT name, COUNT(*) as count,
                   AVG(confidence) as avg_confidence,
                   MIN(timestamp) as first_seen,
                   MAX(timestamp) as last_seen
            FROM detections
            GROUP BY name
            ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # ---- Attendance ----

    def check_in(self, name):
        today = datetime.now().strftime("%Y-%m-%d")
        person = self.get_person(name)
        person_id = person["id"] if person else None
        existing = self.conn.execute(
            "SELECT * FROM attendance WHERE name = ? AND date = ? AND check_out IS NULL",
            (name, today),
        ).fetchone()
        if existing:
            return False  # already checked in
        self.conn.execute(
            "INSERT INTO attendance (person_id, name, check_in, date) VALUES (?, ?, ?, ?)",
            (person_id, name, datetime.now().isoformat(), today),
        )
        self.conn.commit()
        return True

    def check_out(self, name):
        today = datetime.now().strftime("%Y-%m-%d")
        self.conn.execute(
            "UPDATE attendance SET check_out = ? WHERE name = ? AND date = ? AND check_out IS NULL",
            (datetime.now().isoformat(), name, today),
        )
        self.conn.commit()

    def get_attendance(self, date=None):
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        rows = self.conn.execute(
            "SELECT * FROM attendance WHERE date = ? ORDER BY check_in", (date,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_attendance_range(self, start_date, end_date):
        rows = self.conn.execute(
            "SELECT * FROM attendance WHERE date BETWEEN ? AND ? ORDER BY date, check_in",
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
