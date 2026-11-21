"""
Personal data tool -- local SQLite.

This is the user's own private data. It never leaves the machine except as an
already-aggregated number inside the model prompt, and no external source is
ever consulted for it.

Security stance: the model does NOT write SQL. It picks a named query kind
("total", "by_category", "recent", "top") and supplies a period and optional
category. Every SQL statement in this file is a fixed string with bound
parameters. There is no string interpolation of user or model input into SQL,
so there is no injection surface even if the model is confused or adversarial.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS expenses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,          -- ISO YYYY-MM-DD, local date
    category    TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    amount      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);
CREATE INDEX IF NOT EXISTS idx_expenses_category ON expenses(category);
"""

PERIODS = ("today", "this_week", "this_month", "last_month", "this_year", "all_time")


def _period_bounds(period: str) -> tuple[str, str, str]:
    """
    Returns (start_iso, end_iso, spoken_label), inclusive of both ends.

    Uses the machine's LOCAL date. This matters: computing "this month" in UTC
    puts the first and last few hours of every month in the wrong bucket.
    """
    today = date.today()
    if period == "today":
        return today.isoformat(), today.isoformat(), "today"
    if period == "this_week":
        start = today - timedelta(days=today.weekday())
        return start.isoformat(), today.isoformat(), "this week"
    if period == "last_month":
        first_this = today.replace(day=1)
        end = first_this - timedelta(days=1)
        return end.replace(day=1).isoformat(), end.isoformat(), "last month"
    if period == "this_year":
        return today.replace(month=1, day=1).isoformat(), today.isoformat(), "this year"
    if period == "all_time":
        return "0001-01-01", "9999-12-31", "in total"
    return today.replace(day=1).isoformat(), today.isoformat(), "this month"


class PersonalDataTool(Tool):
    name = "query_expenses"
    description = (
        "Query the user's own private expense records stored locally. Use this for any "
        "question about what the user spent, their budget, their categories, or their "
        "recent purchases. This is the only source of truth for the user's personal "
        "finances -- never guess or use general knowledge for these questions."
    )

    def __init__(self, config, db_path: Path):
        self.cfg = config
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query_kind": {
                    "type": "string",
                    "enum": ["total", "by_category", "recent", "top"],
                    "description": "'total' for a sum, 'by_category' for a breakdown, "
                                   "'recent' for the latest individual entries, "
                                   "'top' for the biggest single expenses.",
                },
                "period": {
                    "type": "string",
                    "enum": list(PERIODS),
                    "description": "Time window. Defaults to this_month.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional filter, e.g. 'food', 'transport'.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query_kind"],
        }

    def execute(self, query_kind: str = "total", period: str = "this_month",
                category: str = "", limit: int = 5) -> ToolResult:
        if period not in PERIODS:
            period = "this_month"
        try:
            limit = max(1, min(20, int(limit)))
        except (TypeError, ValueError):
            limit = 5

        start, end, label = _period_bounds(period)
        cat = (category or "").strip().lower()
        params: list = [start, end]
        cat_clause = ""
        if cat:
            cat_clause = " AND LOWER(category) = ?"
            params.append(cat)

        try:
            with self._connect() as conn:
                if query_kind == "by_category":
                    rows = conn.execute(
                        "SELECT category, ROUND(SUM(amount), 2) AS total, COUNT(*) AS n "
                        "FROM expenses WHERE date BETWEEN ? AND ?" + cat_clause +
                        " GROUP BY LOWER(category) ORDER BY total DESC",
                        params,
                    ).fetchall()
                    data = {
                        "period": label, "currency": self.cfg.currency_symbol,
                        "breakdown": [
                            {"category": r["category"], "total": r["total"], "count": r["n"]}
                            for r in rows
                        ],
                    }

                elif query_kind in ("recent", "top"):
                    order = "date DESC, id DESC" if query_kind == "recent" else "amount DESC"
                    rows = conn.execute(
                        "SELECT date, category, description, amount FROM expenses "
                        "WHERE date BETWEEN ? AND ?" + cat_clause +
                        f" ORDER BY {order} LIMIT ?",
                        params + [limit],
                    ).fetchall()
                    data = {
                        "period": label, "currency": self.cfg.currency_symbol,
                        "entries": [dict(r) for r in rows],
                    }

                else:  # total
                    row = conn.execute(
                        "SELECT ROUND(COALESCE(SUM(amount), 0), 2) AS total, COUNT(*) AS n "
                        "FROM expenses WHERE date BETWEEN ? AND ?" + cat_clause,
                        params,
                    ).fetchone()
                    data = {
                        "period": label, "currency": self.cfg.currency_symbol,
                        "total": row["total"], "transaction_count": row["n"],
                    }
                    if cat:
                        data["category"] = cat
        except sqlite3.Error as exc:
            return self._fail(f"Could not read the expense database: {exc}")

        empty = (
            (query_kind == "total" and data.get("transaction_count", 0) == 0)
            or (query_kind == "by_category" and not data.get("breakdown"))
            or (query_kind in ("recent", "top") and not data.get("entries"))
        )
        if empty:
            data["note"] = "No expenses recorded for this period."

        return self._ok(data, query_kind=query_kind, period=period)

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM expenses")

    # -- convenience for seeding and tests ---------------------------------
    def add_expense(self, on: str, category: str, description: str, amount: float) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO expenses (date, category, description, amount) VALUES (?,?,?,?)",
                (on, category, description, float(amount)),
            )
            return int(cur.lastrowid)

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0])


MAX_EXPENSE = 1_000_000


class LogExpenseTool(Tool):
    """Write side of the expense tracker. Kept as its own tool, so a model that
    only meant to *read* can never write by picking the wrong query kind."""

    name = "log_expense"
    description = (
        "Record a new expense the user just told you about, e.g. 'I spent 200 on an auto' "
        "or 'add 450 for lunch'. Only call this when the user clearly states an amount they "
        "spent. Pick a short lowercase category such as food, transport, shopping, "
        "utilities, health, entertainment, travel or other."
    )

    def __init__(self, data_tool: PersonalDataTool):
        self.data = data_tool

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "exclusiveMinimum": 0},
                "category": {"type": "string"},
                "description": {"type": "string", "description": "What it was, in a few words."},
                "date": {"type": "string",
                         "description": "YYYY-MM-DD. Omit for today; use yesterday's date "
                                        "if they said yesterday."},
            },
            "required": ["amount", "category"],
        }

    def execute(self, amount=0, category: str = "other", description: str = "",
                date: str = "", **_) -> ToolResult:
        try:
            value = round(float(amount), 2)
        except (TypeError, ValueError):
            return self._fail("The amount wasn't a number.")
        if not 0 < value <= MAX_EXPENSE:
            return self._fail("That amount looks wrong; it must be above zero.")
        on = (date or "").strip() or _today_iso()
        try:
            _date_cls.fromisoformat(on)
        except ValueError:
            return self._fail("The date must look like 2026-11-21.")
        cat = " ".join((category or "other").lower().split())[:40] or "other"
        desc = " ".join((description or "").split())[:200]
        new_id = self.data.add_expense(on, cat, desc, value)
        return self._ok({"logged": {"id": new_id, "amount": value, "category": cat,
                                    "description": desc, "date": on},
                         "currency": self.data.cfg.currency_symbol})


def _today_iso() -> str:
    return date.today().isoformat()


_date_cls = date
