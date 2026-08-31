#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer

STOP = set("a an and are as at be by for from has have how i in is it of on or our should that the there this to was what when where which who will with you your answer answers correct explicitly final wrapped boxed please concise must say says".split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_line(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = re.sub(r"\b\d{5,}\b", "<NUM>", text)
    text = re.sub(r"[0-9a-f]{12,}", "<ID>", text, flags=re.I)
    return text


def state_lines(state: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in ("url", "action", "thought"):
        value = state.get(key)
        if value:
            parts.append(f"{key.upper()}: {norm_line(value)}")
    tree = state.get("accessibility_tree") or ""
    for line in str(tree).splitlines():
        line = norm_line(line)
        if line:
            parts.append(line)
    return parts


def raw_memory(trajectory: dict[str, Any]) -> str:
    parts = [
        f"GOAL: {norm_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {trajectory.get('outcome', '')}",
    ]
    for state in trajectory.get("states", []):
        parts.append(f"STATE {state.get('state_index', '?')}")
        parts.extend(state_lines(state))
    return "\n".join(parts)


def canonical_memory(trajectory: dict[str, Any]) -> str:
    """Question-independent canonical state memory: retain each normalized line once."""
    seen: set[str] = set()
    parts = [
        f"GOAL: {norm_line(trajectory.get('goal', ''))}",
        f"OUTCOME: {trajectory.get('outcome', '')}",
    ]
    for state in trajectory.get("states", []):
        for line in state_lines(state):
            if line not in seen:
                seen.add(line)
                parts.append(line)
    return "\n".join(parts)


def chunks(text: str, size: int = 4000, overlap: int = 250) -> list[str]:
    if not text:
        return [""]
    output = []
    start = 0
    while start < len(text):
        output.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return output


def make_index(documents: dict[str, str]):
    texts: list[str] = []
    trajectory_ids: list[str] = []
    by_id: dict[str, list[int]] = defaultdict(list)
    for trajectory_id, text in documents.items():
        for chunk in chunks(text):
            by_id[trajectory_id].append(len(texts))
            trajectory_ids.append(trajectory_id)
            texts.append(chunk)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, texts, trajectory_ids, by_id


def retrieve(index, query: str, allowed_ids: list[str], budget: int) -> tuple[str, float]:
    vectorizer, matrix, texts, _, by_id = index
    candidate_indices: list[int] = []
    for trajectory_id in allowed_ids:
        candidate_indices.extend(by_id.get(trajectory_id, []))
    if not candidate_indices:
        return "", 0.0
    started = time.perf_counter()
    query_vector = vectorizer.transform([query])
    scores = (matrix[candidate_indices] @ query_vector.T).toarray().ravel()
    order = np.argsort(-scores)
    selected: list[str] = []
    used = 0
    for offset in order:
        text = texts[candidate_indices[int(offset)]]
        remaining = budget - used
        if remaining <= 0:
            break
        selected.append(text[:remaining])
        used += min(len(text), remaining)
    return "\n\n--- MEMORY CHUNK ---\n\n".join(selected), time.perf_counter() - started


def normalize_answer(text: Any) -> str:
    value = str(text).strip()
    value = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", value, flags=re.I | re.S)
    value = re.sub(r"^(?:final answer|answer)\s*:\s*", "", value, flags=re.I)
    return value.strip()


def scalar_score(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "correct", "is_correct", "result", "accuracy"):
            if key in value:
                return scalar_score(value[key])
    if isinstance(value, (tuple, list)) and value:
        return scalar_score(value[0])
    raise TypeError(f"Cannot convert evaluator output to score: {type(value)} {value!r}")


def import_official_metrics(official_repo: Path):
    sys.path.insert(0, str(official_repo))
    return importlib.import_module("evaluation.qa_eval_metrics")


def make_metric_adapter(metrics_module, question: dict[str, Any]) -> Callable[[str], float] | None:
    name = str(question.get("eval_function", ""))
    if not name or name.startswith("llm_"):
        return None
    fn = getattr(metrics_module, name, None)
    if fn is None or not callable(fn):
        return None
    signature = inspect.signature(fn)
    gold = question.get("answer")
    qtext = question.get("question")

    prediction_names = {
        "prediction", "pred", "response", "model_response", "generated_answer",
        "model_answer", "output", "hypothesis", "candidate",
    }
    reference_names = {
        "answer", "gold", "gold_answer", "reference", "reference_answer",
        "ground_truth", "expected", "expected_answer", "target",
    }
    question_names = {"question", "query", "question_text", "prompt"}

    def invoke(prediction: str):
        args = []
        kwargs = {}
        positional_index = 0
        for parameter in signature.parameters.values():
            pname = parameter.name.lower()
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if pname in prediction_names or any(token in pname for token in ("pred", "response", "output")):
                value = prediction
            elif pname in reference_names or any(token in pname for token in ("gold", "reference")):
                value = gold
            elif pname in question_names:
                value = qtext
            elif parameter.name in question:
                value = question[parameter.name]
            elif pname in question:
                value = question[pname]
            elif positional_index == 0:
                value = prediction
            elif positional_index == 1:
                value = gold
            elif positional_index == 2:
                value = qtext
            elif parameter.default is not inspect._empty:
                positional_index += 1
                continue
            else:
                raise TypeError(f"Unmapped evaluator parameter {parameter.name} for {name}")
            positional_index += 1
            if parameter.kind == parameter.KEYWORD_ONLY:
                kwargs[parameter.name] = value
            else:
                args.append(value)
        return scalar_score(fn(*args, **kwargs))

    # A deterministic evaluator must score its own gold answer as correct.
    try:
        if invoke(str(gold)) < 0.999:
            return None
    except Exception:
        return None
    return invoke


def reader_call(
    *,
    token: str,
    model: str,
    memory: str,
    question: str,
    cache_file: Path,
    max_attempts: int = 8,
) -> tuple[str, float, str]:
    cache_key = hashlib.sha256((model + "\n" + question + "\n" + memory).encode()).hexdigest()
    if cache_file.exists():
        cache = json.loads(cache_file.read_text())
    else:
        cache = {}
    if cache_key in cache:
        item = cache[cache_key]
        return item["answer"], float(item.get("seconds", 0.0)), item.get("model", model)

    endpoint = "https://models.github.ai/inference/chat/completions"
    models = [model]
    for fallback in ("openai/gpt-4.1-mini", "openai/gpt-4o-mini"):
        if fallback not in models:
            models.append(fallback)
    last_error = ""
    for candidate_model in models:
        for attempt in range(max_attempts):
            started = time.perf_counter()
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={
                    "model": candidate_model,
                    "temperature": 0,
                    "max_tokens": 220,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer the question using only the supplied agent memory. "
                                "Return the shortest answer that fully resolves the question. "
                                "Do not explain your reasoning and do not mention the memory."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"AGENT MEMORY:\n{memory}\n\nQUESTION:\n{question}\n\nANSWER:",
                        },
                    ],
                },
                timeout=180,
            )
            elapsed = time.perf_counter() - started
            if response.status_code == 200:
                payload = response.json()
                answer = normalize_answer(payload["choices"][0]["message"]["content"])
                cache[cache_key] = {
                    "answer": answer,
                    "seconds": elapsed,
                    "model": candidate_model,
                }
                cache_file.write_text(json.dumps(cache))
                return answer, elapsed, candidate_model
            last_error = f"{response.status_code}: {response.text[:500]}"
            if response.status_code in (401, 403, 404):
                break
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(min(60, 4 * (2**attempt)))
                continue
            break
    raise RuntimeError(f"Reader API failed: {last_error}")


def stratified_select(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("question_type", "other"))].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
    return selected


def bootstrap_difference(frame: pd.DataFrame, seed: int = 260830) -> dict[str, float]:
    pivot = frame.pivot(index="question_id", columns="method", values="score").dropna()
    diff = (pivot["canonical_unique_lines"] - pivot["raw"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(30_000, len(diff)))].mean(axis=1)
    return {
        "questions": int(len(diff)),
        "paired_mean_gain": float(diff.mean()),
        "ci_95_low": float(np.quantile(boot, 0.025)),
        "ci_95_high": float(np.quantile(boot, 0.975)),
        "compressed_wins": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "raw_wins": int(np.sum(diff < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--out", default="longmemeval_fixed_reader_results")
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--context-chars", type=int, default=8000)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=260830)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {"status": "running"}
    (output / "status.json").write_text(json.dumps(status, indent=2))

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        status = {"status": "blocked", "reason": "GITHUB_TOKEN unavailable"}
        (output / "status.json").write_text(json.dumps(status, indent=2))
        return

    data_root = output / "data"
    snapshot_download(
        repo_id="xiaowu0162/longmemeval-v2",
        repo_type="dataset",
        local_dir=str(data_root),
        allow_patterns=[
            "questions.jsonl",
            "trajectories.jsonl",
            "haystacks/lme_v2_small.json",
        ],
    )
    questions = read_jsonl(data_root / "questions.jsonl")
    haystack = json.loads((data_root / "haystacks/lme_v2_small.json").read_text())
    wanted_ids = {trajectory_id for ids in haystack.values() for trajectory_id in ids}
    trajectories = {
        row["id"]: row
        for row in read_jsonl(data_root / "trajectories.jsonl")
        if row.get("id") in wanted_ids
    }

    raw_docs = {trajectory_id: raw_memory(row) for trajectory_id, row in trajectories.items()}
    canonical_docs = {
        trajectory_id: canonical_memory(row) for trajectory_id, row in trajectories.items()
    }
    raw_chars = sum(map(len, raw_docs.values()))
    canonical_chars = sum(map(len, canonical_docs.values()))
    raw_index = make_index(raw_docs)
    canonical_index = make_index(canonical_docs)

    metrics_module = import_official_metrics(Path(args.official_repo))
    eligible = []
    adapters: dict[str, Callable[[str], float]] = {}
    evaluator_counts: dict[str, int] = defaultdict(int)
    for question in questions:
        if question.get("image") is not None:
            continue
        if question.get("id") not in haystack:
            continue
        adapter = make_metric_adapter(metrics_module, question)
        if adapter is None:
            continue
        question_id = str(question["id"])
        adapters[question_id] = adapter
        eligible.append(question)
        evaluator_counts[str(question.get("eval_function"))] += 1

    selected = stratified_select(eligible, min(args.questions, len(eligible)), args.seed)
    cache_file = output / "reader_cache.json"
    rows = []
    api_error = None
    for question_index, question in enumerate(selected):
        question_id = str(question["id"])
        allowed = haystack[question_id]
        for method, index in (
            ("raw", raw_index),
            ("canonical_unique_lines", canonical_index),
        ):
            memory, retrieval_seconds = retrieve(
                index,
                str(question["question"]),
                allowed,
                args.context_chars,
            )
            try:
                answer, reader_seconds, actual_model = reader_call(
                    token=token,
                    model=args.model,
                    memory=memory,
                    question=str(question["question"]),
                    cache_file=cache_file,
                )
            except Exception as exc:
                api_error = repr(exc)
                break
            try:
                score = float(adapters[question_id](answer))
            except Exception as exc:
                score = np.nan
                evaluator_error = repr(exc)
            else:
                evaluator_error = None
            rows.append(
                {
                    "question_index": question_index,
                    "question_id": question_id,
                    "domain": question.get("domain"),
                    "question_type": question.get("question_type"),
                    "eval_function": question.get("eval_function"),
                    "method": method,
                    "score": score,
                    "prediction": answer,
                    "gold_answer": json.dumps(question.get("answer"), ensure_ascii=False),
                    "retrieved_chars": len(memory),
                    "retrieval_seconds": retrieval_seconds,
                    "reader_seconds": reader_seconds,
                    "reader_model": actual_model,
                    "evaluator_error": evaluator_error,
                }
            )
        pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
        if api_error:
            break

    runs = pd.DataFrame(rows)
    if api_error or runs.empty:
        status = {
            "status": "blocked",
            "reason": api_error or "No completed reader calls",
            "eligible_questions": len(eligible),
            "completed_calls": len(runs),
        }
        (output / "status.json").write_text(json.dumps(status, indent=2))
        (output / "REPORT.md").write_text(
            "# LongMemEval-V2 fixed-reader pilot\n\n"
            f"Status: BLOCKED\n\nReason: `{status['reason']}`\n"
        )
        return

    valid = runs.dropna(subset=["score"])
    paired_ids = set(valid[valid.method == "raw"].question_id) & set(
        valid[valid.method == "canonical_unique_lines"].question_id
    )
    valid = valid[valid.question_id.isin(paired_ids)]
    summary = (
        valid.groupby("method")
        .agg(
            questions=("question_id", "nunique"),
            official_accuracy=("score", "mean"),
            median_retrieval_ms=("retrieval_seconds", lambda x: 1000 * float(np.median(x))),
            median_reader_seconds=("reader_seconds", "median"),
            mean_retrieved_chars=("retrieved_chars", "mean"),
        )
        .reset_index()
    )
    storage = pd.DataFrame(
        [
            {"method": "raw", "stored_chars": raw_chars, "ratio_to_raw": 1.0},
            {
                "method": "canonical_unique_lines",
                "stored_chars": canonical_chars,
                "ratio_to_raw": canonical_chars / raw_chars,
            },
        ]
    )
    summary = summary.merge(storage, on="method", how="left")
    summary.to_csv(output / "summary.csv", index=False)
    paired = bootstrap_difference(valid)
    (output / "paired_bootstrap.json").write_text(json.dumps(paired, indent=2))

    summary_index = summary.set_index("method")
    raw_accuracy = float(summary_index.loc["raw", "official_accuracy"])
    canonical_accuracy = float(
        summary_index.loc["canonical_unique_lines", "official_accuracy"]
    )
    verdict = (
        "GO"
        if canonical_accuracy >= raw_accuracy and canonical_chars / raw_chars <= 0.25
        else "NO-GO"
    )
    metadata = {
        "status": "complete",
        "verdict": verdict,
        "runtime_seconds": time.time() - started,
        "official_questions_total": len(questions),
        "eligible_deterministic_text_questions": len(eligible),
        "selected_questions": len(selected),
        "paired_scored_questions": len(paired_ids),
        "context_char_budget": args.context_chars,
        "requested_reader_model": args.model,
        "actual_reader_models": sorted(valid.reader_model.unique().tolist()),
        "memory_built_before_questions": True,
        "storage_ratio": canonical_chars / raw_chars,
        "paired": paired,
        "scope": (
            "Official LongMemEval-V2 small-tier data and official deterministic evaluators; "
            "paired fixed-reader pilot on a stratified text-only subset. This is not a full "
            "small-tier leaderboard submission."
        ),
    }
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    report = f"""# LongMemEval-V2 fixed-reader paired pilot

## Verdict: {verdict}

The same reader model and prompt answered the same {len(paired_ids)} official questions from two query-independent memory stores.

| Memory | Official deterministic accuracy | Stored history | Median retrieval |
|---|---:|---:|---:|
| Raw | {100*raw_accuracy:.1f}% | 100.0% | {summary_index.loc['raw','median_retrieval_ms']:.2f} ms |
| Canonical unique-line memory | {100*canonical_accuracy:.1f}% | {100*canonical_chars/raw_chars:.1f}% | {summary_index.loc['canonical_unique_lines','median_retrieval_ms']:.2f} ms |

Paired accuracy gain: **{100*paired['paired_mean_gain']:+.1f} points**, 95% bootstrap interval **[{100*paired['ci_95_low']:+.1f}, {100*paired['ci_95_high']:+.1f}]**.

Compressed wins / ties / raw wins: **{paired['compressed_wins']} / {paired['ties']} / {paired['raw_wins']}**.

## Scope

Official small-tier histories and official deterministic evaluators were used. Images and LLM-graded questions were excluded. The reader was `{', '.join(metadata['actual_reader_models'])}` through GitHub Models. The memory representations were built before questions; retrieval was query-conditioned. This is an end-to-end paired pilot rather than a full leaderboard score.
"""
    (output / "REPORT.md").write_text(report)
    print(json.dumps(metadata, sort_keys=True))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
