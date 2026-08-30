from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .memory import Memory, MemoryContextItem, register_memory


_VOLATILE_NUMBER = re.compile(r"\b\d{5,}\b")
_VOLATILE_HEX = re.compile(r"[0-9a-f]{12,}", flags=re.I)
_WHITESPACE = re.compile(r"\s+")


def _normalize_line(value: object) -> str:
    text = _WHITESPACE.sub(" ", str(value)).strip()
    text = _VOLATILE_NUMBER.sub("<NUM>", text)
    text = _VOLATILE_HEX.sub("<ID>", text)
    return text


def _state_lines(state: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("url", "action", "thought"):
        value = state.get(key)
        if value:
            lines.append(f"{key.upper()}: {_normalize_line(value)}")
    tree = state.get("accessibility_tree") or ""
    if tree:
        for raw_line in str(tree).splitlines():
            line = _normalize_line(raw_line)
            if line:
                lines.append(line)
    return lines


def _trajectory_text(trajectory: dict[str, Any], mode: str) -> str:
    prefix = [
        f"GOAL: {_normalize_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {_normalize_line(trajectory.get('outcome', ''))}",
    ]
    if mode == "raw":
        parts = list(prefix)
        for state in trajectory.get("states", []):
            if not isinstance(state, dict):
                continue
            parts.append(f"STATE {state.get('state_index', '?')}")
            parts.extend(_state_lines(state))
        return "\n".join(parts)

    if mode == "unique_lines":
        parts = list(prefix)
        seen: set[str] = set()
        for state in trajectory.get("states", []):
            if not isinstance(state, dict):
                continue
            for line in _state_lines(state):
                if line not in seen:
                    seen.add(line)
                    parts.append(line)
        return "\n".join(parts)

    if mode == "distinction_delta":
        parts = list(prefix)
        previous: set[str] = set()
        seen_events: set[str] = set()
        previous_url: str | None = None
        for state_index, state in enumerate(trajectory.get("states", [])):
            if not isinstance(state, dict):
                continue
            current_lines = _state_lines(state)
            current = set(current_lines)
            url = _normalize_line(state.get("url") or "")
            events: list[str] = []
            if state_index == 0:
                events.extend(current_lines)
            else:
                if url and url != previous_url:
                    events.append(f"URL CHANGED: {previous_url} -> {url}")
                if state.get("action"):
                    events.append(f"ACTION: {_normalize_line(state['action'])}")
                if state.get("thought"):
                    events.append(f"THOUGHT: {_normalize_line(state['thought'])}")
                events.extend(f"APPEARED: {line}" for line in sorted(current - previous))
                events.extend(
                    f"DISAPPEARED: {line}"
                    for line in sorted(previous - current)[:80]
                )
            unique_events: list[str] = []
            for event in events:
                if event and event not in seen_events:
                    seen_events.add(event)
                    unique_events.append(event)
            if unique_events:
                parts.append(f"CHANGE {state.get('state_index', state_index)}")
                parts.extend(unique_events)
            previous = current
            previous_url = url
        return "\n".join(parts)

    raise ValueError(f"Unsupported memory mode: {mode}")


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


@register_memory
class DistinctionTfidfMemory(Memory):
    """Question-independent canonical memory with matched sparse retrieval."""

    memory_type = "distinction_tfidf"

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        self.mode = str(memory_params.get("mode", "unique_lines"))
        self.chunk_chars = int(memory_params.get("chunk_chars", 5000))
        self.chunk_overlap = int(memory_params.get("chunk_overlap", 300))
        self.top_k = int(memory_params.get("top_k", 32))
        if self.mode not in {"raw", "unique_lines", "distinction_delta"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.chunk_chars <= self.chunk_overlap:
            raise ValueError("chunk_chars must exceed chunk_overlap")
        self.vectorizer = HashingVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            n_features=2**18,
            alternate_sign=False,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self.texts: list[str] = []
        self.matrix: sparse.csr_matrix | None = None
        self.raw_characters = 0
        self.stored_characters = 0

    def insert(self, trajectory: dict[str, object]) -> None:
        raw_text = _trajectory_text(trajectory, "raw")
        stored_text = _trajectory_text(trajectory, self.mode)
        self.raw_characters += len(raw_text)
        self.stored_characters += len(stored_text)
        new_chunks = _chunks(stored_text, self.chunk_chars, self.chunk_overlap)
        if not new_chunks:
            return
        encoded = self.vectorizer.transform(new_chunks).tocsr()
        self.matrix = encoded if self.matrix is None else sparse.vstack(
            [self.matrix, encoded], format="csr"
        )
        self.texts.extend(new_chunks)

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        del query_image
        if self.matrix is None or not self.texts:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(-scores)[: self.top_k]
        return [
            {"type": "text", "value": self.texts[int(index)]}
            for index in order
            if self.texts[int(index)].strip()
        ]

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[MemoryContextItem],
    ) -> dict[str, object] | None:
        del query, query_image
        return {
            "mode": self.mode,
            "raw_characters": self.raw_characters,
            "stored_characters": self.stored_characters,
            "storage_ratio": (
                self.stored_characters / self.raw_characters
                if self.raw_characters
                else None
            ),
            "returned_items": len(memory_context),
        }
