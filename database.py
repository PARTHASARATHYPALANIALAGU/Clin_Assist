"""
Database layer for ClinAssist
Handles SQLite operations for sessions, symptom records, conversation turns, and latency logs
"""
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any, Iterator
from contextlib import contextmanager
from config import DATABASE_PATH, STATES


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize database with all required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'greeting',
                asked_fields_json TEXT NOT NULL DEFAULT '[]',
                session_type TEXT NOT NULL DEFAULT 'intake'
            )
        """)

        # Migration for existing databases without asked_fields_json
        cursor.execute("PRAGMA table_info(sessions)")
        session_columns = [row[1] for row in cursor.fetchall()]
        if "asked_fields_json" not in session_columns:
            cursor.execute("ALTER TABLE sessions ADD COLUMN asked_fields_json TEXT NOT NULL DEFAULT '[]'")
        if "session_type" not in session_columns:
            cursor.execute("ALTER TABLE sessions ADD COLUMN session_type TEXT NOT NULL DEFAULT 'intake'")

        # Backfill legacy consult sessions created before session_type existed
        cursor.execute(
            """
            UPDATE sessions
            SET session_type = 'consult'
            WHERE id LIKE 'consult_%'
            """
        )
        
        # Symptom records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symptom_records (
                session_id TEXT PRIMARY KEY,
                chief_complaint TEXT,
                duration TEXT,
                severity INTEGER,
                progression TEXT,
                associated_symptoms_json TEXT,
                affected_body_part TEXT,
                onset_type TEXT,
                aggravating_alleviating_factors TEXT,
                relevant_medical_history TEXT,
                risk_level TEXT,
                risk_reason TEXT,
                recommended_action TEXT,
                summary TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Migration for existing databases
        cursor.execute("PRAGMA table_info(symptom_records)")
        symptom_columns = [row[1] for row in cursor.fetchall()]
        if "aggravating_alleviating_factors" not in symptom_columns:
            cursor.execute("ALTER TABLE symptom_records ADD COLUMN aggravating_alleviating_factors TEXT")
        if "relevant_medical_history" not in symptom_columns:
            cursor.execute("ALTER TABLE symptom_records ADD COLUMN relevant_medical_history TEXT")
        
        # Conversation turns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Latency logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latency_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # TTS Cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tts_cache (
                hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                voice_name TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()


def get_cached_tts(text_hash: str) -> Optional[str]:
    """Check if audio for this hash exists in cache"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM tts_cache WHERE hash = ?", (text_hash,))
        row = cursor.fetchone()
        return row["file_path"] if row else None


def save_tts_to_cache(text_hash: str, text: str, voice_name: str, settings: Dict[str, Any], file_path: str):
    """Save record to TTS cache table"""
    timestamp = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO tts_cache (hash, text, voice_name, settings_json, file_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (text_hash, text, voice_name, json.dumps(settings), str(file_path), timestamp)
        )


def create_session(session_id: str = None, session_type: str = "intake") -> str:
    """Create a new session and return session ID"""
    if not session_id:
        session_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, created_at, state, asked_fields_json, session_type) VALUES (?, ?, ?, ?, ?)",
            (session_id, timestamp, STATES["GREETING"], json.dumps([]), session_type)
        )
        
        # Initialize empty symptom record
        cursor.execute(
            "INSERT INTO symptom_records (session_id) VALUES (?)",
            (session_id,)
        )
    
    return session_id


def update_session_state(session_id: str, state: str):
    """Update session state"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET state = ? WHERE id = ?",
            (state, session_id)
        )


def get_session_state(session_id: str) -> Optional[str]:
    """Get current session state"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT state FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return row["state"] if row else None


def get_session_type(session_id: str) -> Optional[str]:
    """Get session type (intake or consult)"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_type FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return row["session_type"] if row else None


def get_asked_fields(session_id: str) -> List[str]:
    """Get list of fields already asked for this session"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT asked_fields_json FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if not row or not row["asked_fields_json"]:
            return []
        try:
            return json.loads(row["asked_fields_json"])
        except json.JSONDecodeError:
            return []


def set_asked_fields(session_id: str, asked_fields: List[str]):
    """Persist asked fields list for a session"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET asked_fields_json = ? WHERE id = ?",
            (json.dumps(asked_fields), session_id)
        )


def save_turn(session_id: str, role: str, content: str):
    """Save a conversation turn"""
    timestamp = datetime.utcnow().isoformat()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_turns (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
            (session_id, timestamp, role, content)
        )


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve conversation history for a session"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversation_turns WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]


def update_symptom_record(session_id: str, fields: Dict[str, Any]):
    """Update symptom record with new fields"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build dynamic UPDATE query
        update_parts = []
        values = []
        
        for key, value in fields.items():
            if key == "associated_symptoms":
                # Serialize list to JSON
                update_parts.append("associated_symptoms_json = ?")
                values.append(json.dumps(value) if value else None)
            elif key in ["risk_level", "risk_reason", "recommended_action", "summary"]:
                update_parts.append(f"{key} = ?")
                values.append(value)
            elif key in ["chief_complaint", "duration", "severity", "progression", "affected_body_part", "onset_type", "aggravating_alleviating_factors", "relevant_medical_history"]:
                update_parts.append(f"{key} = ?")
                values.append(value)
        
        if update_parts:
            query = f"UPDATE symptom_records SET {', '.join(update_parts)} WHERE session_id = ?"
            values.append(session_id)
            cursor.execute(query, values)


def get_symptom_record(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve symptom record for a session"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM symptom_records WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Convert row to dictionary
        record = dict(row)
        
        # Parse associated_symptoms JSON (resilient to legacy malformed data)
        raw_associated = record.get("associated_symptoms_json")
        if raw_associated:
            try:
                parsed = json.loads(raw_associated)
                if isinstance(parsed, list):
                    record["associated_symptoms"] = parsed
                elif isinstance(parsed, str) and parsed.strip():
                    record["associated_symptoms"] = [parsed.strip()]
                else:
                    record["associated_symptoms"] = None
            except (json.JSONDecodeError, TypeError):
                record["associated_symptoms"] = [str(raw_associated).strip()] if str(raw_associated).strip() else None
        else:
            record["associated_symptoms"] = None
        
        del record["associated_symptoms_json"]
        
        return record


def log_latency(session_id: str, operation_type: str, latency_ms: float):
    """Log latency for an operation"""
    timestamp = datetime.utcnow().isoformat()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO latency_logs (session_id, timestamp, operation_type, latency_ms) VALUES (?, ?, ?, ?)",
            (session_id, timestamp, operation_type, latency_ms)
        )


def session_exists(session_id: str) -> bool:
    """Check if a session exists"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        return cursor.fetchone() is not None


def get_session_export_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Get all data for session export"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        if not session:
            return None
        
        # Get conversation history
        history = get_session_history(session_id)
        
        # Get symptom record
        symptom_record = get_symptom_record(session_id)
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "state": session["state"],
            "conversation_history": history,
            "symptom_record": symptom_record
        }


def get_recent_sessions(limit: int = 10, session_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve a list of recent sessions for history management"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
            SELECT s.id, s.created_at, s.state, s.session_type, sr.chief_complaint, sr.risk_level
            FROM sessions s
            LEFT JOIN symptom_records sr ON s.id = sr.session_id
        """
        params = []
        if session_type:
            query += " WHERE s.session_type = ?"
            params.append(session_type)

        query += " ORDER BY s.created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_session(session_id: str) -> bool:
    """Delete a session and all its associated data"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check if session exists first
        cursor.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        if not cursor.fetchone():
            return False
            
        # Delete from related tables first (foreign key constraints)
        cursor.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM latency_logs WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM symptom_records WHERE session_id = ?", (session_id,))
        
        # Finally delete the session itself
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        return True
