import json
import os
import sqlite3

DB_PATH = os.path.expanduser("~/.eda_assistant_sessions.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            preferences TEXT NOT NULL,
            active_goal TEXT,
            interaction_history TEXT NOT NULL DEFAULT '[]',
            last_file TEXT,
            profile_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def save_session(session_id, preferences, interaction_history,
                 active_goal=None, last_file=None, profile_json=None):
    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT updated_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        data = {
            "preferences": json.dumps(preferences),
            "active_goal": active_goal,
            "interaction_history": json.dumps(interaction_history, default=str),
            "last_file": last_file,
            "profile_json": json.dumps(profile_json, default=str) if profile_json else None,
        }
        if existing:
            conn.execute("""
                UPDATE sessions SET
                    preferences=:preferences, active_goal=:active_goal,
                    interaction_history=:interaction_history, last_file=:last_file,
                    profile_json=:profile_json, updated_at=datetime('now')
                WHERE session_id=:session_id
            """, {**data, "session_id": session_id})
        else:
            conn.execute("""
                INSERT INTO sessions
                    (session_id, preferences, active_goal, interaction_history,
                     last_file, profile_json)
                VALUES
                    (:session_id, :preferences, :active_goal, :interaction_history,
                     :last_file, :profile_json)
            """, {**data, "session_id": session_id})
        conn.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def load_session(session_id):
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT preferences, active_goal, interaction_history, last_file, profile_json "
            "FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        preferences, active_goal, hist_json, last_file, profile_json = row
        result = {
            "preferences": json.loads(preferences),
            "active_goal": active_goal,
            "interaction_history": json.loads(hist_json),
            "last_file": last_file,
        }
        if profile_json:
            try:
                result["profile_json"] = json.loads(profile_json)
            except (json.JSONDecodeError, TypeError):
                result["profile_json"] = None
        return result
    finally:
        conn.close()


def list_sessions():
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT session_id, last_file, updated_at "
            "FROM sessions ORDER BY updated_at DESC LIMIT 20"
        ).fetchall()
        return [{"id": r[0], "file": r[1], "updated": r[2]} for r in rows]
    finally:
        conn.close()


def delete_session(session_id):
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()
