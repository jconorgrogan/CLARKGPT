from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .memory import Memory, MemoryContextItem, register_memory


_WHITESPACE = re.compile(r"\s+")
_LEADING_BID = re.compile(r"^\s*\[[A-Za-z]?\d+\]\s*")
_ACTION_BID = re.compile(
    r"(?P<fn>\b(?:click|fill|hover|select_option|press)\()"
    r"(?P<q>['\"])[A-Za-z]?\d+(?P=q)"
)
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_VALUE = re.compile(r"\bvalue=['\"]([^'\"]*)['\"]")
_ROLE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\b")
_OPTION = re.compile(
    r"(?ms)^\s*([A-H])\.\s*(.*?)(?=^\s*[A-H]\.\s|\n\s*(?:Put|Your final|Please|Return|Answer|Give)\b|\Z)"
)
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")
_MAGENTO_TAB_NOISE = re.compile(
    r"^The information in this tab has been changed\. This tab contains invalid data\. "
    r"Please resolve this before saving\. Loading\.\.\.\s*",
    flags=re.I,
)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._@/-]*")

_MEANINGFUL_ROLES = {
    "RootWebArea", "heading", "link", "button", "tab", "columnheader",
    "gridcell", "textbox", "combobox", "option", "checkbox", "radio",
    "StaticText", "menuitem", "treeitem", "cell", "rowheader", "strong",
    "LabelText", "status", "alert", "dialog", "listitem", "paragraph",
}
_STRUCTURAL_ROLES = {"RootWebArea", "heading", "tab", "columnheader", "rowheader", "dialog", "status", "alert"}
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "if", "in", "into", "is", "it", "its", "may", "of", "on", "or", "our",
    "should", "that", "the", "their", "there", "these", "they", "this", "to", "using",
    "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your", "option", "correct", "environment", "page", "website",
}


def _norm(value: object) -> str:
    return _WHITESPACE.sub(" ", str(value)).strip()


def _clean_visible_text(value: object) -> str:
    text = _PRIVATE_USE.sub(" ", str(value))
    text = _MAGENTO_TAB_NOISE.sub("", text)
    return _norm(text).strip(" -|,")


def _is_contentful(text: str) -> bool:
    if not text:
        return False
    alnum = sum(ch.isalnum() for ch in text)
    return alnum >= 1 and not all(ch.isdigit() for ch in text) if len(text) <= 2 else alnum >= 2


def _canonical_url(value: object) -> str:
    text = _norm(value)
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    if not parsed.scheme or not parsed.netloc:
        return text
    query_keys = sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)})
    query = "&".join(query_keys)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def _normalize_action(value: object) -> str:
    text = _norm(value)
    return _ACTION_BID.sub(lambda match: f"{match.group('fn')}<BID>", text)


def _tokens(value: str) -> set[str]:
    return {t for t in _TOKEN.findall(value.lower()) if len(t) > 1 and t not in _STOP}


def _raw_state_lines(state: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("url", "action", "thought"):
        value = state.get(key)
        if value:
            lines.append(f"{key.upper()}: {_norm(value)}")
    tree = state.get("accessibility_tree") or ""
    for raw_line in str(tree).splitlines():
        line = _norm(raw_line)
        if line:
            lines.append(line)
    return lines


def _raw_trajectory_text(trajectory: dict[str, Any]) -> str:
    parts = [
        f"GOAL: {_norm(trajectory.get('goal', ''))}",
        f"OUTCOME: {_norm(trajectory.get('outcome', ''))}",
    ]
    for state in trajectory.get("states", []):
        if not isinstance(state, dict):
            continue
        parts.append(f"STATE {state.get('state_index', '?')}")
        parts.extend(_raw_state_lines(state))
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


def _parse_fact(raw_line: object) -> tuple[str, str] | None:
    text = _LEADING_BID.sub("", str(raw_line)).strip()
    text = _norm(text)
    if not text:
        return None
    role_match = _ROLE.match(text)
    if not role_match:
        return None
    role = role_match.group(1)
    if role not in _MEANINGFUL_ROLES:
        return None

    quoted = [left or right for left, right in _QUOTED.findall(text)]
    name = _clean_visible_text(quoted[0]) if quoted else ""
    value_match = _VALUE.search(text)
    value = _clean_visible_text(value_match.group(1)) if value_match else ""
    if not _is_contentful(name) and not _is_contentful(value):
        return None

    attrs: list[str] = []
    lower = text.lower().replace("'", "").replace('"', "")
    for key in ("selected=true", "selected=false", "expanded=true", "expanded=false", "checked=true", "checked=false"):
        if key in lower:
            attrs.append(key)
    if "haspopup=" in lower:
        popup = re.search(r"hasPopup=['\"]?([^,'\" ]+)", text, flags=re.I)
        if popup:
            attrs.append(f"popup={popup.group(1)}")

    pieces = [role]
    if name:
        pieces.append(name)
    if value and value != name:
        pieces.append(f"value={value}")
    pieces.extend(attrs)
    return " | ".join(pieces), role


@dataclass
class _State:
    trajectory_key: str
    trajectory_position: int
    state_index: str
    goal: str
    outcome: str
    page: str
    url: str
    action: str
    facts: list[int]
    roles: list[str]
    search_text: str
    token_set: set[str]


@register_memory
class DistinctionGraphMemory(Memory):
    """Raw, state-bundle, or incidence-preserving quotient memory."""

    memory_type = "distinction_graph"

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)
        self.mode = str(memory_params.get("mode", "incidence_bundle"))
        self.chunk_chars = int(memory_params.get("chunk_chars", 5000))
        self.chunk_overlap = int(memory_params.get("chunk_overlap", 300))
        self.top_k = int(memory_params.get("top_k", 20))
        self.render_chars = int(memory_params.get("render_chars", 1450))
        self.max_facts_per_bundle = int(memory_params.get("max_facts_per_bundle", 48))
        if self.mode not in {"raw", "state_bundle", "incidence_bundle"}:
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
        self.raw_texts: list[str] = []
        self.raw_matrix: sparse.csr_matrix | None = None
        self.fact_texts: list[str] = []
        self.fact_to_id: dict[str, int] = {}
        self.fact_supports: list[list[int]] = []
        self.states: list[_State] = []
        self.trajectory_states: dict[str, list[int]] = {}
        self.fact_matrix: sparse.csr_matrix | None = None
        self.state_matrix: sparse.csr_matrix | None = None
        self._built = False
        self.raw_characters = 0
        self.stored_characters = 0
        self.incidence_edges = 0
        self.trajectory_count = 0

    def insert(self, trajectory: dict[str, object]) -> None:
        raw_text = _raw_trajectory_text(trajectory)
        self.raw_characters += len(raw_text)
        self.trajectory_count += 1
        if self.mode == "raw":
            self.raw_texts.extend(_chunks(raw_text, self.chunk_chars, self.chunk_overlap))
            self.stored_characters += len(raw_text)
            self._built = False
            return

        trajectory_key = str(
            trajectory.get("trajectory_id")
            or trajectory.get("id")
            or trajectory.get("task_id")
            or f"trajectory_{self.trajectory_count - 1}"
        )
        goal = _norm(trajectory.get("goal", ""))
        outcome = _norm(trajectory.get("outcome", ""))
        state_ids: list[int] = []
        for position, raw_state in enumerate(trajectory.get("states", [])):
            if not isinstance(raw_state, dict):
                continue
            facts: list[int] = []
            roles: list[str] = []
            seen_local: set[str] = set()
            page = ""
            tree = raw_state.get("accessibility_tree") or ""
            for raw_line in str(tree).splitlines():
                parsed = _parse_fact(raw_line)
                if parsed is None:
                    continue
                fact, role = parsed
                if fact in seen_local:
                    continue
                seen_local.add(fact)
                if role == "RootWebArea" and not page:
                    page = fact.split(" | ", 1)[-1]
                fact_id = self.fact_to_id.get(fact)
                if fact_id is None:
                    fact_id = len(self.fact_texts)
                    self.fact_to_id[fact] = fact_id
                    self.fact_texts.append(fact)
                    self.fact_supports.append([])
                    self.stored_characters += len(fact) + 2
                facts.append(fact_id)
                roles.append(role)

            url = _canonical_url(raw_state.get("url") or "")
            action = _normalize_action(raw_state.get("action") or "")
            state_index = str(raw_state.get("state_index", position))
            fact_strings = [self.fact_texts[fid] for fid in facts]
            search_text = "\n".join(
                part for part in (
                    f"PAGE {page}" if page else "",
                    f"URL {url}" if url else "",
                    f"ACTION {action}" if action else "",
                    f"GOAL {goal}" if goal else "",
                    f"OUTCOME {outcome}" if outcome else "",
                    *fact_strings,
                ) if part
            )
            state_id = len(self.states)
            self.states.append(_State(
                trajectory_key=trajectory_key,
                trajectory_position=position,
                state_index=state_index,
                goal=goal,
                outcome=outcome,
                page=page,
                url=url,
                action=action,
                facts=facts,
                roles=roles,
                search_text=search_text,
                token_set=_tokens(search_text),
            ))
            state_ids.append(state_id)
            for fact_id in facts:
                self.fact_supports[fact_id].append(state_id)
            self.incidence_edges += len(facts)
            meta = f"{trajectory_key}|{position}|{state_index}|{page}|{url}|{action}|{outcome}"
            pointer_chars = sum(len(str(fid)) + 1 for fid in facts)
            self.stored_characters += len(meta) + pointer_chars + 2
        self.trajectory_states[trajectory_key] = state_ids
        self._built = False

    def _ensure_built(self) -> None:
        if self._built:
            return
        if self.mode == "raw":
            self.raw_matrix = self.vectorizer.transform(self.raw_texts).tocsr() if self.raw_texts else None
        else:
            self.fact_matrix = self.vectorizer.transform(self.fact_texts).tocsr() if self.fact_texts else None
            state_docs = [state.search_text for state in self.states]
            self.state_matrix = self.vectorizer.transform(state_docs).tocsr() if state_docs else None
        self._built = True

    @staticmethod
    def _query_variants(query: str) -> tuple[str, list[str]]:
        matches = list(_OPTION.finditer(query))
        if len(matches) < 3:
            return query, []
        stem = query[: matches[0].start()].strip()
        return stem or query, [_norm(match.group(2)) for match in matches]

    @staticmethod
    def _priority(scores: np.ndarray) -> np.ndarray:
        if scores.ndim == 1:
            return scores
        if scores.shape[1] == 1:
            return scores[:, 0]
        stem = scores[:, 0]
        option_scores = scores[:, 1:]
        option_max = option_scores.max(axis=1)
        second = np.partition(option_scores, -2, axis=1)[:, -2] if option_scores.shape[1] > 1 else np.zeros_like(option_max)
        contrast = np.maximum(0.0, option_max - second)
        return 0.22 * stem + 0.58 * option_max + 0.20 * contrast

    def _score_graph(self, query: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.state_matrix is not None
        stem, options = self._query_variants(query)
        variants = [stem] + options if options else [query]
        qmat = self.vectorizer.transform(variants)
        state_scores_all = (self.state_matrix @ qmat.T).toarray()
        direct = self._priority(state_scores_all)
        if self.mode == "state_bundle" or self.fact_matrix is None or not self.fact_texts:
            return direct, np.zeros(len(self.fact_texts), dtype=np.float32), state_scores_all
        fact_scores_all = (self.fact_matrix @ qmat.T).toarray()
        fact_priority = self._priority(fact_scores_all)
        support = np.zeros(len(self.states), dtype=np.float32)
        for state_id, state in enumerate(self.states):
            if not state.facts:
                continue
            vals = fact_priority[np.asarray(state.facts, dtype=np.int64)]
            if vals.size:
                top = np.sort(vals)[-min(4, vals.size):]
                support[state_id] = float(top[-1] + 0.35 * top.mean())
        return 0.66 * support + 0.34 * direct, fact_priority, state_scores_all

    def _select_states(self, query: str, base: np.ndarray, state_scores_all: np.ndarray) -> list[int]:
        if base.size == 0:
            return []
        candidate_count = min(len(base), max(160, self.top_k * 12))
        candidates = list(np.argsort(-base, kind="stable")[:candidate_count])
        if state_scores_all.ndim == 2 and state_scores_all.shape[1] > 1:
            for col in range(state_scores_all.shape[1]):
                candidates.extend(np.argsort(-state_scores_all[:, col], kind="stable")[:24].tolist())
        candidates = list(dict.fromkeys(int(x) for x in candidates))
        query_tokens = _tokens(query)
        selected: list[int] = []
        covered: set[str] = set()
        used_pages: set[str] = set()
        used_trajectories: set[str] = set()
        for _ in range(min(self.top_k, len(candidates))):
            best_id: int | None = None
            best_score = -math.inf
            for state_id in candidates:
                if state_id in selected:
                    continue
                state = self.states[state_id]
                overlap = state.token_set & query_tokens
                coverage_gain = len(overlap - covered) / max(1, len(query_tokens))
                page_gain = 0.045 if state.page and state.page not in used_pages else 0.0
                trajectory_gain = 0.025 if state.trajectory_key not in used_trajectories else 0.0
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        len(state.token_set & self.states[other].token_set)
                        / max(1, len(state.token_set | self.states[other].token_set))
                        for other in selected
                    )
                score = float(base[state_id]) + 0.20 * coverage_gain + page_gain + trajectory_gain - 0.12 * redundancy
                if score > best_score:
                    best_score = score
                    best_id = state_id
            if best_id is None:
                break
            selected.append(best_id)
            chosen = self.states[best_id]
            covered.update(chosen.token_set & query_tokens)
            if chosen.page:
                used_pages.add(chosen.page)
            used_trajectories.add(chosen.trajectory_key)
        return selected

    def _adjacent_action(self, state: _State, offset: int) -> str:
        ids = self.trajectory_states.get(state.trajectory_key, [])
        pos = state.trajectory_position + offset
        if pos < 0 or pos >= len(ids):
            return ""
        return self.states[ids[pos]].action

    def _render_state(self, state_id: int, fact_priority: np.ndarray, query: str) -> str:
        state = self.states[state_id]
        query_tokens = _tokens(query)
        local_scores: list[tuple[float, int]] = []
        for idx, fact_id in enumerate(state.facts):
            score = float(fact_priority[fact_id]) if fact_priority.size else 0.0
            lexical = len(_tokens(self.fact_texts[fact_id]) & query_tokens) / max(1, len(query_tokens))
            structural = 0.025 if state.roles[idx] in _STRUCTURAL_ROLES else 0.0
            local_scores.append((score + 0.20 * lexical + structural, idx))
        hit_indices = [idx for score, idx in sorted(local_scores, reverse=True)[:14] if score > 0]
        keep: set[int] = set()
        for idx in hit_indices:
            for neighbor in range(max(0, idx - 2), min(len(state.facts), idx + 3)):
                keep.add(neighbor)
        for idx, role in enumerate(state.roles):
            if role in _STRUCTURAL_ROLES:
                keep.add(idx)
        if not keep:
            keep.update(range(min(len(state.facts), self.max_facts_per_bundle)))
        ordered = sorted(keep)
        if len(ordered) > self.max_facts_per_bundle:
            rank = {idx: score for score, idx in local_scores}
            ordered = sorted(sorted(ordered, key=lambda i: rank.get(i, 0.0), reverse=True)[: self.max_facts_per_bundle])
        lines = [f"SUPPORT STATE trajectory={state.trajectory_key} position={state.trajectory_position} state={state.state_index}"]
        if state.page:
            lines.append(f"PAGE: {state.page}")
        if state.url:
            lines.append(f"URL: {state.url}")
        if state.outcome:
            lines.append(f"TRAJECTORY OUTCOME: {state.outcome}")
        prev_action = self._adjacent_action(state, -1)
        next_action = self._adjacent_action(state, 1)
        if prev_action:
            lines.append(f"PREVIOUS ACTION: {prev_action}")
        if state.action:
            lines.append(f"ACTION: {state.action}")
        if next_action:
            lines.append(f"NEXT ACTION: {next_action}")
        lines.append("OBSERVED TOGETHER:")
        lines.extend(f"- {self.fact_texts[state.facts[idx]]}" for idx in ordered)
        text = "\n".join(lines)
        if len(text) > self.render_chars:
            text = text[: self.render_chars].rsplit("\n", 1)[0]
        return text

    def query(self, query: str, query_image: str | None = None) -> list[MemoryContextItem]:
        del query_image
        self._ensure_built()
        if self.mode == "raw":
            if self.raw_matrix is None or not self.raw_texts:
                return []
            query_vector = self.vectorizer.transform([query])
            scores = (self.raw_matrix @ query_vector.T).toarray().ravel()
            order = np.argsort(-scores, kind="stable")[: self.top_k]
            return [
                {"type": "text", "value": self.raw_texts[int(index)]}
                for index in order
                if self.raw_texts[int(index)].strip()
            ]
        if self.state_matrix is None or not self.states:
            return []
        base, fact_priority, state_scores_all = self._score_graph(query)
        selected = self._select_states(query, base, state_scores_all)
        return [
            {"type": "text", "value": self._render_state(state_id, fact_priority, query)}
            for state_id in selected
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
            "storage_ratio": self.stored_characters / self.raw_characters if self.raw_characters else None,
            "returned_items": len(memory_context),
            "trajectory_count": self.trajectory_count,
            "state_count": len(self.states),
            "unique_fact_count": len(self.fact_texts),
            "incidence_edges": self.incidence_edges,
        }
