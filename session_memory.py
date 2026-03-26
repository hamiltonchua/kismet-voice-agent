import sqlite3
import threading
from pathlib import Path


DATA_DIR = Path(__file__).parent / 'data'
SESSION_DB_PATH = DATA_DIR / 'session_memory.db'
_session_db_lock = threading.Lock()


def init_session_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _session_db_lock, sqlite3.connect(SESSION_DB_PATH) as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        conn.commit()


def load_recent_session_messages(limit: int) -> list[dict[str, str]]:
    try:
        with _session_db_lock, sqlite3.connect(SESSION_DB_PATH) as conn:
            rows = conn.execute(
                'SELECT role, content FROM messages ORDER BY id DESC LIMIT ?',
                (limit,),
            ).fetchall()
        messages = [{'role': role, 'content': content} for role, content in reversed(rows)]
        print(f'[Memory] Restored {len(messages)} session messages from SQLite')
        return messages
    except Exception as e:
        print(f'[Memory] Failed loading SQLite session memory: {type(e).__name__}: {e}')
        return []


def persist_session_message(role: str, content: str) -> None:
    if not content.strip():
        return
    try:
        with _session_db_lock, sqlite3.connect(SESSION_DB_PATH) as conn:
            conn.execute(
                'INSERT INTO messages (role, content) VALUES (?, ?)',
                (role, content),
            )
            conn.commit()
    except Exception as e:
        print(f'[Memory] Failed persisting SQLite session message: {type(e).__name__}: {e}')


def remove_last_session_message(role: str | None = None) -> None:
    try:
        with _session_db_lock, sqlite3.connect(SESSION_DB_PATH) as conn:
            if role:
                row = conn.execute(
                    'SELECT id FROM messages WHERE role = ? ORDER BY id DESC LIMIT 1',
                    (role,),
                ).fetchone()
            else:
                row = conn.execute('SELECT id FROM messages ORDER BY id DESC LIMIT 1').fetchone()
            if row:
                conn.execute('DELETE FROM messages WHERE id = ?', (row[0],))
                conn.commit()
    except Exception as e:
        print(f'[Memory] Failed removing SQLite session message: {type(e).__name__}: {e}')


def clear_session_memory() -> None:
    try:
        with _session_db_lock, sqlite3.connect(SESSION_DB_PATH) as conn:
            conn.execute('DELETE FROM messages')
            conn.commit()
        print('[Memory] Cleared SQLite session memory')
    except Exception as e:
        print(f'[Memory] Failed clearing SQLite session memory: {type(e).__name__}: {e}')
