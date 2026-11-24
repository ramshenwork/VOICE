"""
Knowledge base: your own documents (project notes, specs, plans) dropped into
the knowledge/ folder as .md or .txt files.

Files are split into paragraph-sized chunks and ranked with BM25, a classic
keyword-relevance score. No embedding model, no network, no extra dependency;
for a personal folder of documents it is fast and good enough. The index is
rebuilt automatically when any file is added, changed or removed.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

EXTENSIONS = {".md", ".txt", ".markdown"}
CHUNK_CHARS = 900
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "it", "this", "that", "with", "as", "at", "by", "be", "my", "me", "i", "we", "you",
    "what", "how", "about", "do", "does", "did", "from", "our", "your",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


@dataclass
class Chunk:
    source: str
    text: str
    tokens: Counter
    length: int


class KnowledgeBase:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._sig: tuple | None = None
        self._chunks: list[Chunk] = []
        self._df: Counter = Counter()
        self._avg = 1.0

    def _signature(self) -> tuple:
        if not self.folder.is_dir():
            return ()
        return tuple(sorted(
            (str(p), p.stat().st_mtime, p.stat().st_size)
            for p in self.folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS))

    def _refresh(self) -> None:
        sig = self._signature()
        if sig == self._sig:
            return
        chunks: list[Chunk] = []
        for path_str, _, _ in sig:
            path = Path(path_str)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(path.relative_to(self.folder))
            for piece in self._split(text):
                toks = _tokens(piece)
                if toks:
                    chunks.append(Chunk(rel, piece, Counter(toks), len(toks)))
        df: Counter = Counter()
        for c in chunks:
            df.update(set(c.tokens))
        self._chunks, self._df, self._sig = chunks, df, sig
        self._avg = (sum(c.length for c in chunks) / len(chunks)) if chunks else 1.0
        logger.info("knowledge base: %d files, %d chunks", len(sig), len(chunks))

    @staticmethod
    def _split(text: str) -> list[str]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        out, cur = [], ""
        for p in paras:
            if cur and len(cur) + len(p) > CHUNK_CHARS:
                out.append(cur)
                cur = p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
            while len(cur) > CHUNK_CHARS * 2:
                out.append(cur[:CHUNK_CHARS])
                cur = cur[CHUNK_CHARS:]
        if cur:
            out.append(cur)
        return out

    def files(self) -> list[str]:
        self._refresh()
        return sorted({c.source for c in self._chunks})

    def search(self, query: str, limit: int = 4) -> list[dict]:
        self._refresh()
        q = _tokens(query)
        if not q or not self._chunks:
            return []
        n = len(self._chunks)
        k1, b = 1.5, 0.75
        scored = []
        for c in self._chunks:
            score = 0.0
            for t in q:
                tf = c.tokens.get(t, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * c.length / self._avg))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"source": c.source, "text": c.text, "score": round(s, 2)}
                for s, c in scored[:limit]]
