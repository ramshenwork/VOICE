"""
Memory and notes tools.

`memory` is the assistant's own understanding of you: small durable facts it
decides are worth keeping. `notes` is your notebook: things you explicitly
asked to have written down. They are separate tools so the model never
confuses "I noticed you like X" with "you told me to write X down".
"""

from __future__ import annotations

from memory.store import MemoryStore

from .base import Tool, ToolResult


class MemoryTool(Tool):
    name = "memory"
    description = (
        "Your long-term memory about the user. Use action 'remember' whenever the user "
        "tells you something durable about themselves, their people, preferences, routines, "
        "places, or projects (for example 'my office is in Gachibowli', 'I'm building a smart "
        "helmet'), or explicitly says 'remember that'. Use 'recall' to look something up "
        "that is not already in your context. Use 'forget' when asked to forget something. "
        "Do not store trivia, passwords, card numbers or one-off requests."
    )

    def __init__(self, store: MemoryStore):
        self.store = store

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["remember", "recall", "forget"]},
                "fact": {"type": "string",
                         "description": "For 'remember': one short self-contained fact, "
                                        "written in the third person, e.g. 'User's office "
                                        "is in Gachibowli'."},
                "category": {"type": "string",
                             "description": "For 'remember': e.g. personal, preference, "
                                            "place, project, people, routine."},
                "query": {"type": "string",
                          "description": "For 'recall' or 'forget': words to search for."},
            },
            "required": ["action"],
        }

    def execute(self, action: str = "recall", fact: str = "", category: str = "general",
                query: str = "", **_) -> ToolResult:
        if action == "remember":
            if not fact.strip():
                return self._fail("Nothing to remember: 'fact' was empty.")
            return self._ok(self.store.remember(fact, category))
        if action == "forget":
            if not query.strip():
                return self._fail("Say what to forget in 'query'.")
            n = self.store.forget(query=query)
            return self._ok({"forgotten": n})
        found = self.store.search_memories(query)
        return self._ok({"matches": found} if found else
                        {"matches": [], "note": "Nothing remembered about that."})


class NotesTool(Tool):
    name = "notes"
    description = (
        "The user's personal notebook. Use action 'add' when the user asks you to note, "
        "write down, jot or save something; put their words in 'body' faithfully (lightly "
        "tidied, never invented). Use 'search' to find notes about a topic, 'list' for the "
        "latest notes, 'delete' to remove one by id after confirming which."
    )

    def __init__(self, store: MemoryStore):
        self.store = store

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "search", "list", "delete"]},
                "body": {"type": "string", "description": "For 'add': the note text."},
                "title": {"type": "string", "description": "For 'add': a short title."},
                "tags": {"type": "string",
                         "description": "For 'add': comma-separated topics, e.g. 'helmet, ideas'."},
                "query": {"type": "string", "description": "For 'search': words to look for."},
                "note_id": {"type": "integer", "description": "For 'delete'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["action"],
        }

    def execute(self, action: str = "list", body: str = "", title: str = "", tags: str = "",
                query: str = "", note_id: int | None = None, limit: int = 5, **_) -> ToolResult:
        try:
            limit = max(1, min(20, int(limit)))
        except (TypeError, ValueError):
            limit = 5
        if action == "add":
            if not body.strip():
                return self._fail("The note was empty.")
            return self._ok({"saved": self.store.add_note(body, title, tags)})
        if action == "delete":
            if note_id is None:
                return self._fail("Which note? Give its id.")
            n = self.store.delete_note(int(note_id))
            return self._ok({"deleted": n}) if n else self._fail(f"No note with id {note_id}.")
        if action == "search":
            found = self.store.search_notes(query, limit)
            return self._ok({"notes": found} if found else
                            {"notes": [], "note": "No notes match that."})
        notes = self.store.list_notes(limit)
        return self._ok({"notes": notes} if notes else {"notes": [], "note": "No notes yet."})


class PreferencesTool(Tool):
    name = "preferences"
    description = (
        "Standing rules the user wants you to always follow. Use action 'add' when they "
        "correct how you behave or state a lasting rule: 'from now on...', 'always...', "
        "'never...', 'don't do that again', 'too long, keep it shorter', 'stop calling me "
        "sir'. Write the rule as a short imperative, e.g. 'Keep replies under two "
        "sentences unless asked to explain'. Use 'list' when asked what your rules are, "
        "'remove' when they cancel one. Facts about the user go in memory, not here."
    )

    def __init__(self, store: MemoryStore):
        self.store = store

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "remove"]},
                "rule": {"type": "string", "description": "For 'add': the rule."},
                "query": {"type": "string",
                          "description": "For 'remove': words from the rule to cancel."},
                "rule_id": {"type": "integer", "description": "For 'remove', if known."},
            },
            "required": ["action"],
        }

    def execute(self, action: str = "list", rule: str = "", query: str = "",
                rule_id: int | None = None, **_) -> ToolResult:
        if action == "add":
            if not rule.strip():
                return self._fail("The rule was empty.")
            return self._ok(self.store.add_rule(rule))
        if action == "remove":
            if rule_id is None and not query.strip():
                return self._fail("Say which rule to remove.")
            n = self.store.remove_rule(rule_id=rule_id, query=query)
            return self._ok({"removed": n}) if n else self._fail("No rule matched that.")
        rules = self.store.rules()
        return self._ok({"rules": rules} if rules else {"rules": [], "note": "No rules yet."})
