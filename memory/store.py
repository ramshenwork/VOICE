"""
Long-term memory, stored locally in SQLite.

Three kinds of thing live here, and they are deliberately kept apart:

  memories      short facts about the user ("works on a helmet project",
                "prefers to leave office by 6:30"). Put in front of the model
                on every turn, so the assistant *knows* you without having to
                look anything up.
  notes         things the user explicitly asked to have written down. Never
                injected automatically; read back only when asked.
  conversation  a log of plain spoken turns, so a restart picks up where the
                last session left off instead of meeting you for the first time.

Same security stance as the expense tool: fixed SQL strings, bound parameters,
the model never writes SQL.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL,
    category    TEXT    NOT NULL DEFAULT 'general',
    fact        TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL,
    title       TEXT    NOT NULL DEFAULT '',
    body        TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS conversation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL
);
"""

MAX_TEXT = 2000

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "about", "my",
    "me", "i", "is", "was", "what", "did", "do", "any", "all", "that", "this",
    "with", "note", "notes", "it", "you", "your", "from",
}


def _clip(text: str) -> str:
    return " ".join((text or "").split())[:MAX_TEXT]


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- memories ----------------------------------------------------------
    def remember(self, fact: str, category: str = "general") -> dict:
        fact = _clip(fact)
        category = (_clip(category) or "general").lower()[:40]
        with self._connect() as conn:
            dup = conn.execute("SELECT id FROM memories WHERE LOWER(fact) = LOWER(?)",
                               (fact,)).fetchone()
            if dup:
                return {"id": dup["id"], "fact": fact, "duplicate": True}
            cur = conn.execute(
                "INSERT INTO memories (created_at, category, fact) VALUES (?,?,?)",
                (time.time(), category, fact))
            return {"id": int(cur.lastrowid), "fact": fact, "category": category}

    def memories(self, limit: int = 40) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, category, fact FROM memories ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def search_memories(self, query: str, limit: int = 10) -> list[dict]:
        return self._search("memories", ("fact", "category"), query, limit,
                            "id, category, fact")

    def forget(self, memory_id: int | None = None, query: str = "") -> int:
        with self._connect() as conn:
            if memory_id is not None:
                return conn.execute("DELETE FROM memories WHERE id = ?",
                                    (int(memory_id),)).rowcount
            if query.strip():
                return conn.execute("DELETE FROM memories WHERE LOWER(fact) LIKE ?",
                                    (f"%{query.strip().lower()}%",)).rowcount
        return 0

    # -- notes -------------------------------------------------------------
    def add_note(self, body: str, title: str = "", tags: str = "") -> dict:
        body = _clip(body)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO notes (created_at, title, body, tags) VALUES (?,?,?,?)",
                (time.time(), _clip(title)[:120], body, _clip(tags)[:200].lower()))
            return {"id": int(cur.lastrowid), "title": title, "body": body}

    def list_notes(self, limit: int = 5) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, title, body, tags FROM notes "
                "ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
        return [self._note(r) for r in rows]

    def search_notes(self, query: str, limit: int = 5) -> list[dict]:
        rows = self._search("notes", ("title", "body", "tags"), query, limit,
                            "id, created_at, title, body, tags", raw=True)
        return [self._note(r) for r in rows]

    def delete_note(self, note_id: int) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM notes WHERE id = ?", (int(note_id),)).rowcount

    @staticmethod
    def _note(r) -> dict:
        d = dict(r)
        d["created"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.pop("created_at")))
        return d

    # -- conversation log --------------------------------------------------
    def log_turn(self, role: str, content: str) -> None:
        if role not in ("user", "assistant") or not (content or "").strip():
            return
        with self._connect() as conn:
            conn.execute("INSERT INTO conversation (created_at, role, content) VALUES (?,?,?)",
                         (time.time(), role, _clip(content)))

    def recent_turns(self, n: int = 6) -> list[dict]:
        """Last n turns, oldest first, always starting on a user turn."""
        if n <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM conversation ORDER BY id DESC LIMIT ?",
                (int(n),)).fetchall()
        turns = [dict(r) for r in reversed(rows)]
        while turns and turns[0]["role"] != "user":
            turns.pop(0)
        return turns

    # -- shared ------------------------------------------------------------
    def _search(self, table: str, columns: tuple[str, ...], query: str, limit: int,
                select: str, raw: bool = False):
        # `table`, `columns` and `select` are compile-time constants from this
        # file, never model input. Only the search words are parameters.
        words = [w.strip(".,!?;:'\"") for w in (query or "").lower().split()]
        words = [w for w in words if len(w) > 1 and w not in _STOPWORDS][:6]
        if not words:
            return []
        clauses, params = [], []
        for w in words:
            clauses.append("(" + " OR ".join(f"LOWER({c}) LIKE ?" for c in columns) + ")")
            params.extend([f"%{w}%"] * len(columns))
        sql = (f"SELECT {select} FROM {table} WHERE " + " OR ".join(clauses) +
               " ORDER BY id DESC LIMIT ?")
        with self._connect() as conn:
            rows = conn.execute(sql, params + [max(1, int(limit))]).fetchall()
        return rows if raw else [dict(r) for r in rows]
