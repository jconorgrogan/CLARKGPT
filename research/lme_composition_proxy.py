from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

import longmemeval_v2_distinction_proxy as base

OUT = Path(os.environ.get("OUT", "longmemeval_composition_proxy_results"))
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "data"

LEADING_BID = re.compile(r"^\[[A-Za-z]?\d+\]\s*")
ACTION_BID = re.compile(r"(?P<fn>\b(?:click|fill|hover|select_option|press)\()(?P<q>['\"])[A-Za-z]?\d+(?P=q)")
QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def canonical_url(value: object) -> str:
    text = base.norm_line(value)
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except ValueError:
        return text
    if not parsed.scheme or not parsed.netloc:
        return text
    keys = sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)})
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "&".join(keys), ""))


def normalize_action(value: object) -> str:
    text = base.norm_line(value)
    return ACTION_BID.sub(lambda match: f"{match.group('fn')}<BID>", text)


def semantic_tree_line(raw: object) -> str | None:
    text = LEADING_BID.sub("", base.norm_line(raw))
    if not text:
        return None
    quoted = [left or right for left, right in QUOTED.findall(text)]
    has_name = any(value.strip() for value in quoted)
    has_value = bool(re.search(r"\bvalue=['\"][^'\"]+['\"]", text))
    if not has_name and not has_value:
        return None
    return ACTION_BID.sub(lambda match: f"{match.group('fn')}<BID>", text)


def semantic_state(raw_state: dict) -> dict[str, object]:
    lines: list[str] = []
    seen: set[str] = set()
    for raw in str(raw_state.get("accessibility_tree") or "").splitlines():
        line = semantic_tree_line(raw)
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return {
        "index": raw_state.get("state_index", "?"),
        "url": canonical_url(raw_state.get("url") or ""),
        "action": normalize_action(raw_state.get("action") or ""),
        "thought": base.norm_line(raw_state.get("thought") or ""),
        "lines": lines,
    }


def prefix(trajectory: dict) -> list[str]:
    return [
        f"GOAL: {base.norm_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {base.norm_line(trajectory.get('outcome', ''))}",
    ]


def anchored_unique_memory(trajectory: dict) -> str:
    """Deduplicate content while retaining the state/history labels lost by the quotient."""
    parts = prefix(trajectory)
    seen_content: set[str] = set()
    previous_url = ""
    for position, raw_state in enumerate(trajectory.get("states", [])):
        if not isinstance(raw_state, dict):
            continue
        state = semantic_state(raw_state)
        events: list[str] = []
        url = str(state["url"])
        if url and url != previous_url:
            events.append(f"URL: {url}")
        if state["action"]:
            events.append(f"ACTION: {state['action']}")
        if state["thought"]:
            events.append(f"THOUGHT: {state['thought']}")
        for line in state["lines"]:
            line = str(line)
            if line not in seen_content:
                seen_content.add(line)
                events.append(f"FIRST SEEN: {line}")
        if events:
            parts.append(f"STATE {state['index']} POSITION {position}")
            parts.extend(events)
        previous_url = url or previous_url
    return "\n".join(parts)


def semantic_delta_memory(trajectory: dict) -> str:
    """Canonical semantic state differences with explicit composition/history labels."""
    parts = prefix(trajectory)
    previous: set[str] = set()
    previous_url = ""
    for position, raw_state in enumerate(trajectory.get("states", [])):
        if not isinstance(raw_state, dict):
            continue
        state = semantic_state(raw_state)
        current = {str(line) for line in state["lines"]}
        events: list[str] = []
        url = str(state["url"])
        if url and url != previous_url:
            events.append(f"URL: {url}")
        if state["action"]:
            events.append(f"ACTION: {state['action']}")
        if state["thought"]:
            events.append(f"THOUGHT: {state['thought']}")
        if position == 0:
            events.extend(f"PRESENT: {line}" for line in sorted(current))
        else:
            events.extend(f"APPEARED: {line}" for line in sorted(current - previous))
            events.extend(f"DISAPPEARED: {line}" for line in sorted(previous - current))
        if events:
            parts.append(f"STATE {state['index']} POSITION {position}")
            parts.extend(events)
        previous = current
        previous_url = url or previous_url
    return "\n".join(parts)


def page_segment_memory(trajectory: dict) -> str:
    """Union semantic observations within each consecutive page segment."""
    states = [semantic_state(state) for state in trajectory.get("states", []) if isinstance(state, dict)]
    parts = prefix(trajectory)
    segments: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_url: str | None = None
    for state in states:
        url = str(state["url"])
        key = url or current_url or "UNKNOWN"
        if current and key != current_url:
            segments.append(current)
            current = []
        current.append(state)
        current_url = key
    if current:
        segments.append(current)

    for segment_index, segment in enumerate(segments):
        first, last = segment[0], segment[-1]
        url = str(first["url"] or last["url"] or "UNKNOWN")
        parts.append(f"PAGE {segment_index} STATES {first['index']}-{last['index']} URL: {url}")
        seen_actions: set[str] = set()
        seen_thoughts: set[str] = set()
        seen_lines: set[str] = set()
        for state in segment:
            action, thought = str(state["action"]), str(state["thought"])
            if action and action not in seen_actions:
                seen_actions.add(action)
                parts.append(f"ACTION: {action}")
            if thought and thought not in seen_thoughts:
                seen_thoughts.add(thought)
                parts.append(f"THOUGHT: {thought}")
            for value in state["lines"]:
                line = str(value)
                if line not in seen_lines:
                    seen_lines.add(line)
                    parts.append(line)
    return "\n".join(parts)


def line_span_memory(trajectory: dict) -> str:
    """Store each semantic line once, labeled by its temporal support interval and count."""
    states = [semantic_state(state) for state in trajectory.get("states", []) if isinstance(state, dict)]
    parts = prefix(trajectory)
    occurrences: dict[str, list[int]] = defaultdict(list)
    for position, state in enumerate(states):
        if state["url"]:
            occurrences[f"URL: {state['url']}"] .append(position)
        if state["action"]:
            parts.append(f"STATE {state['index']} ACTION: {state['action']}")
        if state["thought"]:
            parts.append(f"STATE {state['index']} THOUGHT: {state['thought']}")
        for value in state["lines"]:
            occurrences[str(value)].append(position)
    for line, positions in occurrences.items():
        first, last = positions[0], positions[-1]
        label = f"ONLY STATE {first}" if first == last else f"STATES {first}-{last} COUNT {len(positions)}"
        parts.append(f"{label}: {line}")
    return "\n".join(parts)


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    started = time.time()
    snapshot_download(
        repo_id="xiaowu0162/longmemeval-v2",
        repo_type="dataset",
        local_dir=str(DATA),
        allow_patterns=["questions.jsonl", "trajectories.jsonl", "haystacks/lme_v2_small.json"],
    )
    questions = list(base.read_jsonl(DATA / "questions.jsonl"))
    haystacks = json.loads((DATA / "haystacks/lme_v2_small.json").read_text())
    wanted = {trajectory_id for ids in haystacks.values() for trajectory_id in ids}
    trajectories = {
        row["id"]: row
        for row in base.read_jsonl(DATA / "trajectories.jsonl")
        if row.get("id") in wanted
    }
    if wanted - set(trajectories):
        raise RuntimeError(f"Missing trajectories: {len(wanted - set(trajectories))}")

    raw = {tid: base.full_memory(tr) for tid, tr in trajectories.items()}
    methods = {
        "raw": raw,
        "unique_lines": {tid: base.unique_line_memory(tr) for tid, tr in trajectories.items()},
        "distinction_delta": {tid: base.distinction_memory(tr) for tid, tr in trajectories.items()},
        "anchored_unique": {tid: anchored_unique_memory(tr) for tid, tr in trajectories.items()},
        "semantic_delta": {tid: semantic_delta_memory(tr) for tid, tr in trajectories.items()},
        "page_segments": {tid: page_segment_memory(tr) for tid, tr in trajectories.items()},
        "line_spans": {tid: line_span_memory(tr) for tid, tr in trajectories.items()},
    }
    methods["uniform_semantic_delta_matched"] = {
        tid: base.uniform_sample(raw[tid], len(methods["semantic_delta"][tid])) for tid in raw
    }

    raw_chars = sum(map(len, raw.values()))
    storage = pd.DataFrame([
        {
            "method": name,
            "stored_chars": sum(map(len, docs.values())),
            "ratio_to_raw": sum(map(len, docs.values())) / raw_chars,
            "trajectories": len(docs),
        }
        for name, docs in methods.items()
    ])
    storage.to_csv(OUT / "storage.csv", index=False)
    indexes = {name: base.make_index(docs) for name, docs in methods.items()}

    raw_token_sets = {tid: set(base.answer_tokens(text)) for tid, text in raw.items()}
    eligible = []
    for question in questions:
        if question.get("image") is not None or str(question.get("eval_function", "")).startswith("llm_"):
            continue
        tokens = base.answer_tokens(question.get("answer", ""))
        if not tokens or len(str(question.get("answer", ""))) > 300:
            continue
        union: set[str] = set()
        for trajectory_id in haystacks[question["id"]]:
            union |= raw_token_sets.get(trajectory_id, set())
        oracle = len(set(tokens) & union) / len(set(tokens))
        if oracle >= 0.60:
            eligible.append((question, tokens, oracle))

    rows = []
    budgets = [4000, 8000, 12000, 16000, 24000, 32000]
    for question, tokens, oracle in eligible:
        phrase = base.normalized_phrase(question.get("answer", ""))
        for method, index in indexes.items():
            for budget in budgets:
                context, latency = base.retrieve(index, question["question"], haystacks[question["id"]], budget)
                context_tokens = set(base.answer_tokens(context))
                rows.append({
                    "question_id": question["id"],
                    "domain": question["domain"],
                    "question_type": question["question_type"],
                    "method": method,
                    "context_char_budget": budget,
                    "answer_token_recall": len(set(tokens) & context_tokens) / len(set(tokens)),
                    "exact_answer_phrase_hit": int(bool(phrase) and phrase in base.normalized_phrase(context)),
                    "oracle_lexical_recall": oracle,
                    "query_latency_seconds": latency,
                })
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    summary = (
        runs.groupby(["method", "context_char_budget"])
        .agg(
            questions=("question_id", "nunique"),
            mean_answer_token_recall=("answer_token_recall", "mean"),
            exact_answer_phrase_hit_rate=("exact_answer_phrase_hit", "mean"),
            median_query_latency_ms=("query_latency_seconds", lambda values: 1000 * float(np.median(values))),
        )
        .reset_index()
        .merge(storage[["method", "ratio_to_raw", "stored_chars"]], on="method", how="left")
    )
    summary.to_csv(OUT / "summary.csv", index=False)

    paired_rows = []
    for budget in budgets:
        pivot = runs[runs.context_char_budget == budget].pivot(
            index="question_id", columns="method", values="answer_token_recall"
        )
        for method in methods:
            if method == "raw":
                continue
            delta = (pivot[method] - pivot["raw"]).dropna().to_numpy(float)
            low, high = bootstrap_ci(delta, 260831 + budget + len(method))
            paired_rows.append({
                "method": method,
                "context_char_budget": budget,
                "questions": len(delta),
                "mean_delta_vs_raw": delta.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_vs_raw.csv", index=False)

    type_summary = (
        runs[runs.context_char_budget == 16000]
        .groupby(["method", "question_type"])
        .agg(
            questions=("question_id", "nunique"),
            mean_answer_token_recall=("answer_token_recall", "mean"),
            exact_answer_phrase_hit_rate=("exact_answer_phrase_hit", "mean"),
        )
        .reset_index()
    )
    type_summary.to_csv(OUT / "by_question_type_16k.csv", index=False)

    metadata = {
        "runtime_seconds": time.time() - started,
        "official_questions": len(questions),
        "official_small_haystack_trajectories": len(trajectories),
        "eligible_lexically_supported_text_questions": len(eligible),
        "raw_stored_chars": raw_chars,
        "headline_budget_chars": 16000,
        "note": "Question-independent compression built before questions; retrieval/evidence proxy, not reader accuracy.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("HEADLINE_16K")
    print(summary[summary.context_char_budget == 16000].sort_values("mean_answer_token_recall", ascending=False).to_string(index=False))
    print("PAIRED_16K")
    print(paired[paired.context_char_budget == 16000].sort_values("mean_delta_vs_raw", ascending=False).to_string(index=False))
    print("STORAGE")
    print(storage.sort_values("ratio_to_raw").to_string(index=False))
    print("META", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
