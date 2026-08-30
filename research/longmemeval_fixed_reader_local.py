#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer

WEB_SYSTEM = (
    "You are an experienced colleague in a web browsing environment that has "
    "a customized magento-based shopping website, a customized magento-based "
    "shopping admin cms website, as well as a customized forum website based "
    "on reddit/postmill. Answer based on your memory of the environment. "
    "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
    "Do not guess. Never attempt to guess an answer if you are not sure. "
    "If you believe the question's construction/premise is wrong, provide an "
    "explanation in \\boxed{} explaining why the question is flawed."
)
ENTERPRISE_SYSTEM = (
    "You are an experienced colleague working in a customized ServiceNow "
    "environment. Answer based on your memory of the environment. "
    "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
    "Do not guess. Never attempt to guess an answer if you are not sure. "
    "If you believe the question's construction/premise is wrong, provide an "
    "explanation in \\boxed{} explaining why the question is flawed."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def norm_line(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"\b\d{5,}\b", "<NUM>", text)
    text = re.sub(r"[0-9a-f]{12,}", "<ID>", text, flags=re.I)
    return text


def state_lines(state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ["url", "action", "thought"]:
        value = state.get(key)
        if value:
            out.append(f"{key.upper()}: {norm_line(value)}")
    tree = state.get("accessibility_tree") or ""
    for line in str(tree).splitlines():
        normalized = norm_line(line)
        if normalized:
            out.append(normalized)
    return out


def raw_memory(trajectory: dict[str, Any]) -> str:
    out = [
        f"GOAL: {norm_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {norm_line(trajectory.get('outcome', ''))}",
    ]
    for state in trajectory.get("states", []):
        out.append(f"STATE {state.get('state_index', '?')}")
        out.extend(state_lines(state))
    return "\n".join(out)


def unique_line_memory(trajectory: dict[str, Any]) -> str:
    seen: set[str] = set()
    out = [
        f"GOAL: {norm_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {norm_line(trajectory.get('outcome', ''))}",
    ]
    for state in trajectory.get("states", []):
        for line in state_lines(state):
            if line not in seen:
                seen.add(line)
                out.append(line)
    return "\n".join(out)


def chunks(text: str, size: int = 4000, overlap: int = 250) -> list[str]:
    if not text:
        return [""]
    result: list[str] = []
    start = 0
    while start < len(text):
        result.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return result


class SparseIndex:
    def __init__(self, documents: dict[str, str]) -> None:
        texts: list[str] = []
        self.trajectory_ids: list[str] = []
        self.by_trajectory: dict[str, list[int]] = defaultdict(list)
        for trajectory_id, document in documents.items():
            for chunk in chunks(document):
                self.by_trajectory[trajectory_id].append(len(texts))
                self.trajectory_ids.append(trajectory_id)
                texts.append(chunk)
        self.texts = texts
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=100000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, allowed_ids: list[str], char_budget: int) -> tuple[str, float]:
        candidates: list[int] = []
        for trajectory_id in allowed_ids:
            candidates.extend(self.by_trajectory.get(trajectory_id, []))
        if not candidates:
            return "", 0.0
        started = time.perf_counter()
        q = self.vectorizer.transform([query])
        scores = (self.matrix[candidates] @ q.T).toarray().ravel()
        order = np.argsort(-scores)
        selected: list[str] = []
        used = 0
        for position in order:
            remaining = char_budget - used
            if remaining <= 0:
                break
            text = self.texts[candidates[int(position)]]
            selected.append(text[:remaining])
            used += min(len(text), remaining)
        return "\n".join(selected), time.perf_counter() - started


def choose_questions(questions: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    eligible = [
        q for q in questions
        if q.get("image") is None
        and not str(q.get("eval_function", "")).startswith("llm_")
        and isinstance(q.get("question"), str)
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for q in eligible:
        grouped[(str(q["domain"]), str(q["question_type"]))].append(q)
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    keys = sorted(grouped)
    while len(selected) < min(limit, len(eligible)):
        progressed = False
        for key in keys:
            rows = grouped[key]
            if rows:
                selected.append(rows.pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


def call_reader(base_url: str, model: str, system_prompt: str, context: str, question: str, timeout: int) -> tuple[str, float]:
    user = f"### Memory context:\n{context if context else '(empty)'}\n\n### Question to answer:\n{question}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 128,
        "stream": False,
    }
    error: Exception | None = None
    for attempt in range(4):
        try:
            started = time.perf_counter()
            response = requests.post(
                base_url.rstrip("/") + "/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            return str(text).strip(), time.perf_counter() - started
        except Exception as exc:
            error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Reader request failed: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="longmemeval_fixed_reader_results")
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3.5-0.8B-Q4_0")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--context-chars", type=int, default=16000)
    parser.add_argument("--selection-seed", type=int, default=20260830)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    data_root = output / "data"
    snapshot_download(
        repo_id="xiaowu0162/longmemeval-v2",
        repo_type="dataset",
        local_dir=str(data_root),
        allow_patterns=["questions.jsonl", "trajectories.jsonl", "haystacks/lme_v2_small.json"],
    )

    sys.path.insert(0, str(Path(args.official_repo).resolve()))
    from evaluation.qa_eval_metrics import eval_from_spec, extract_boxed_answer, is_unknown, score_to_bool

    questions = read_jsonl(data_root / "questions.jsonl")
    trajectories_all = read_jsonl(data_root / "trajectories.jsonl")
    haystack = json.loads((data_root / "haystacks/lme_v2_small.json").read_text())
    selected = choose_questions(questions, args.limit, args.selection_seed)
    selected_ids = {q["id"] for q in selected}
    required_trajectory_ids = {tid for qid in selected_ids for tid in haystack[qid]}
    trajectories = {
        str(row["id"]): row
        for row in trajectories_all
        if row.get("id") in required_trajectory_ids
    }
    missing = required_trajectory_ids - set(trajectories)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} trajectories")

    raw_documents = {tid: raw_memory(tr) for tid, tr in trajectories.items()}
    unique_documents = {tid: unique_line_memory(tr) for tid, tr in trajectories.items()}
    methods = {
        "raw_sparse": SparseIndex(raw_documents),
        "unique_line_canonical": SparseIndex(unique_documents),
    }
    raw_chars = sum(map(len, raw_documents.values()))
    storage = pd.DataFrame([
        {"method": "raw_sparse", "stored_chars": raw_chars, "ratio_to_raw": 1.0},
        {
            "method": "unique_line_canonical",
            "stored_chars": sum(map(len, unique_documents.values())),
            "ratio_to_raw": sum(map(len, unique_documents.values())) / raw_chars,
        },
    ])
    storage.to_csv(output / "storage.csv", index=False)
    (output / "selected_question_ids.json").write_text(json.dumps([q["id"] for q in selected], indent=2))

    rows: list[dict[str, Any]] = []
    checkpoint = output / "runs.csv"
    if checkpoint.exists():
        rows = pd.read_csv(checkpoint).to_dict("records")
    completed = {(str(r["question_id"]), str(r["method"])) for r in rows}

    started_all = time.time()
    for question_index, q in enumerate(selected, start=1):
        system_prompt = WEB_SYSTEM if q["domain"] == "web" else ENTERPRISE_SYSTEM
        for method, index in methods.items():
            key = (str(q["id"]), method)
            if key in completed:
                continue
            context, retrieval_seconds = index.retrieve(
                str(q["question"]),
                list(haystack[q["id"]]),
                args.context_chars,
            )
            response, reader_seconds = call_reader(
                args.base_url,
                args.model,
                system_prompt,
                context,
                str(q["question"]),
                timeout=600,
            )
            parsed = extract_boxed_answer(response)
            score = False
            error = ""
            try:
                score = score_to_bool(eval_from_spec(q["eval_function"], parsed, q["answer"]))
                if is_unknown(parsed):
                    score = False
            except Exception as exc:
                error = repr(exc)
            rows.append({
                "question_id": q["id"],
                "question_index": question_index,
                "domain": q["domain"],
                "question_type": q["question_type"],
                "eval_function": q["eval_function"],
                "method": method,
                "correct": int(score),
                "response_raw": response,
                "response_parsed_boxed": parsed,
                "gold_answer": q["answer"],
                "context_chars": len(context),
                "retrieval_seconds": retrieval_seconds,
                "reader_seconds": reader_seconds,
                "score_error": error,
            })
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            print(
                json.dumps({
                    "done": len(rows),
                    "total": len(selected) * len(methods),
                    "question_id": q["id"],
                    "method": method,
                    "correct": int(score),
                    "reader_seconds": round(reader_seconds, 3),
                }),
                flush=True,
            )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("method")
        .agg(
            questions=("question_id", "nunique"),
            accuracy=("correct", "mean"),
            median_reader_seconds=("reader_seconds", "median"),
            median_retrieval_ms=("retrieval_seconds", lambda x: 1000 * float(np.median(x))),
            mean_context_chars=("context_chars", "mean"),
        )
        .reset_index()
        .merge(storage, on="method", how="left")
    )
    summary.to_csv(output / "summary.csv", index=False)

    pivot = runs.pivot(index="question_id", columns="method", values="correct").dropna()
    paired_difference = (
        pivot["unique_line_canonical"] - pivot["raw_sparse"]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(20260830)
    boot = paired_difference[
        rng.integers(0, len(paired_difference), size=(20000, len(paired_difference)))
    ].mean(axis=1)
    raw_only = int(((pivot["raw_sparse"] == 1) & (pivot["unique_line_canonical"] == 0)).sum())
    unique_only = int(((pivot["raw_sparse"] == 0) & (pivot["unique_line_canonical"] == 1)).sum())
    paired = {
        "questions": int(len(pivot)),
        "accuracy_gain_unique_minus_raw": float(paired_difference.mean()),
        "bootstrap_95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_95_high": float(np.quantile(boot, 0.975)),
        "unique_only_correct": unique_only,
        "raw_only_correct": raw_only,
        "reader_model": args.model,
        "context_chars": args.context_chars,
        "selection_seed": args.selection_seed,
        "elapsed_seconds": time.time() - started_all,
        "scope": "Official LME-V2 small data, exact benchmark system/user prompt shape, exact deterministic evaluators; local Qwen3.5-0.8B substitute reader, not official Qwen3.5-9B leaderboard reader.",
    }
    (output / "paired_result.json").write_text(json.dumps(paired, indent=2))
    by_type = (
        runs.groupby(["method", "domain", "question_type"])
        .agg(questions=("question_id", "nunique"), accuracy=("correct", "mean"))
        .reset_index()
    )
    by_type.to_csv(output / "by_type.csv", index=False)
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("PAIRED", json.dumps(paired, sort_keys=True))


if __name__ == "__main__":
    main()
