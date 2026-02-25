import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = "chat_history.db"

class ChatHistoryManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT, -- JSON string
                created_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()

    def create_session(self, user_id: str, title: str = "New Chat") -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, title, now, now)
        )
        conn.commit()
        conn.close()
        return session_id

    def add_message(self, session_id: str, role: str, content: str, citations: List[Dict] = None):
        now = datetime.now().isoformat()
        citations_json = json.dumps(citations) if citations else None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Insert message
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, citations_json, now)
        )
        
        # Update session timestamp and potentially title (if first user message)
        cursor.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        
        # Auto-update title if it's the first user message and title is generic
        if role == "user":
            cursor.execute("SELECT count(*) FROM messages WHERE session_id = ?", (session_id,))
            count = cursor.fetchone()[0]
            if count <= 2: # System msg + User msg
                new_title = content[:30] + "..." if len(content) > 30 else content
                cursor.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))

        conn.commit()
        conn.close()

    def get_sessions(self, user_id: str) -> List[Dict]:
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_messages(self, session_id: str) -> List[Dict]:
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            msg = dict(row)
            if msg['citations']:
                msg['citations'] = json.loads(msg['citations'])
            messages.append(msg)
            
        return messages

    def delete_session(self, session_id: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
