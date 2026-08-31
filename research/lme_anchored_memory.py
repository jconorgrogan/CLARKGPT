from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .memory import Memory, MemoryContextItem, register_memory


_VOLATILE_NUMBER = re.compile(r"\b\d{5,}\b")
_VOLATILE_HEX = re.compile(r"[0-9a-f]{12,}", flags=re.I)
_WHITESPACE = re.compile(r"\s+")
_LEADING_BID = re.compile(r"^\[[A-Za-z]?\d+\]\s*")
_ACTION_BID = re.compile(
    r"(?P<fn>\b(?:click|fill|hover|select_option|press)\()"
    r"(?P<q>['\"])[A-Za-z]?\d+(?P=q)"
)
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _normalize_line(value: object) -> str:
    text = _WHITESPACE.sub(" ", str(value)).strip()
    text = _VOLATILE_NUMBER.sub("<NUM>", text)
    text = _VOLATILE_HEX.sub("<ID>", text)
    return text


def _normalize_action(value: object) -> str:
    text = _normalize_line(value)
    return _ACTION_BID.sub(lambda match: f"{match.group('fn')}<BID>", text)


def _canonical_url(value: object) -> str:
    text = _normalize_line(value)
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    if not parsed.scheme or not parsed.netloc:
        return text
    query_keys = sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)})
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "&".join(query_keys), ""))


def _raw_state_lines(state: dict[str, Any]) -> list[str]:
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


def _semantic_tree_line(raw: object) -> str | None:
    text = _LEADING_BID.sub("", _normalize_line(raw))
    if not text:
        return None
    quoted = [left or right for left, right in _QUOTED.findall(text)]
    has_name = any(value.strip() for value in quoted)
    has_value = bool(re.search(r"\bvalue=['\"][^'\"]+['\"]", text))
    if not has_name and not has_value:
        return None
    return _ACTION_BID.sub(lambda match: f"{match.group('fn')}<BID>", text)


def _semantic_state(state: dict[str, Any]) -> dict[str, object]:
    lines: list[str] = []
    seen: set[str] = set()
    tree = state.get("accessibility_tree") or ""
    for raw_line in str(tree).splitlines():
        line = _semantic_tree_line(raw_line)
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return {
        "index": state.get("state_index", "?"),
        "url": _canonical_url(state.get("url") or ""),
        "action": _normalize_action(state.get("action") or ""),
        "thought": _normalize_line(state.get("thought") or ""),
        "lines": lines,
    }


def _raw_trajectory_text(trajectory: dict[str, Any]) -> str:
    parts = [
        f"GOAL: {_normalize_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {_normalize_line(trajectory.get('outcome', ''))}",
    ]
    for state in trajectory.get("states", []):
        if not isinstance(state, dict):
            continue
        parts.append(f"STATE {state.get('state_index', '?')}")
        parts.extend(_raw_state_lines(state))
    return "\n".join(parts)


def _anchored_unique_text(trajectory: dict[str, Any]) -> str:
    """Deduplicate semantic content while retaining state/history incidence."""
    parts = [
        f"GOAL: {_normalize_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {_normalize_line(trajectory.get('outcome', ''))}",
    ]
    seen_content: set[str] = set()
    previous_url = ""
    for position, raw_state in enumerate(trajectory.get("states", [])):
        if not isinstance(raw_state, dict):
            continue
        state = _semantic_state(raw_state)
        events: list[str] = []
        url = str(state["url"])
        if url and url != previous_url:
            events.append(f"URL: {url}")
        action = str(state["action"])
        thought = str(state["thought"])
        if action:
            events.append(f"ACTION: {action}")
        if thought:
            events.append(f"THOUGHT: {thought}")
        for value in state["lines"]:
            line = str(value)
            if line not in seen_content:
                seen_content.add(line)
                events.append(f"FIRST SEEN: {line}")
        if events:
            parts.append(f"STATE {state['index']} POSITION {position}")
            parts.extend(events)
        previous_url = url or previous_url
    return "\n".join(parts)


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
    """Matched sparse retrieval over raw or composition-preserving memory."""

    memory_type = "distinction_tfidf"

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        self.mode = str(memory_params.get("mode", "anchored_unique"))
        self.chunk_chars = int(memory_params.get("chunk_chars", 5000))
        self.chunk_overlap = int(memory_params.get("chunk_overlap", 300))
        self.top_k = int(memory_params.get("top_k", 20))
        if self.mode not in {"raw", "anchored_unique"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.chunk_chars <= self.chunk_overlap:
            raise ValueError("chunk_chars must exceed chunk_overlap")
        self.vectorizer = HashingVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
            dtype=np.float32,
        )
        self.texts: list[str] = []
        self.matrix: sparse.csr_matrix | None = None
        self.raw_characters = 0
        self.stored_characters = 0

    def insert(self, trajectory: dict[str, object]) -> None:
        raw_text = _raw_trajectory_text(trajectory)
        stored_text = raw_text if self.mode == "raw" else _anchored_unique_text(trajectory)
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
        order = np.argsort(-scores, kind="stable")[: self.top_k]
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
