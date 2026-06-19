import os
import sqlite3
import datetime
import logging

logger = logging.getLogger(__name__)

AUDIT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "database", "audit_log.db"
)

def init_audit_db():
    conn = sqlite3.connect(AUDIT_DB_PATH)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            question TEXT,
            answer TEXT,
            blocked INTEGER DEFAULT 0,
            block_reason TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_query(question: str, answer: str = None, blocked: bool = False, block_reason: str = None):
    """Log every query attempt — accountability trail"""
    try:
        init_audit_db()
        conn = sqlite3.connect(AUDIT_DB_PATH)
        conn.execute('''
            INSERT INTO audit_log(question, answer, blocked, blocked_reason)
            VALUES(?, ?, ?, ?)
        ''', (question, answer, int(blocked), block_reason))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Audit logging failed: {e}")