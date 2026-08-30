#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

import longmemeval_fixed_reader_local as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="longmemeval_fixed_reader_parallel_results")
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="Qwen3.5-0.8B-Q4_0")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--context-chars", type=int, default=16000)
    parser.add_argument("--selection-seed", type=int, default=20260830)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("workers must be positive")

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
    from evaluation.qa_eval_metrics import (
        eval_from_spec,
        extract_boxed_answer,
        is_unknown,
        score_to_bool,
    )

    questions = base.read_jsonl(data_root / "questions.jsonl")
    trajectories_all = base.read_jsonl(data_root / "trajectories.jsonl")
    haystack = json.loads((data_root / "haystacks/lme_v2_small.json").read_text())
    selected = base.choose_questions(questions, args.limit, args.selection_seed)
    selected_ids = {str(q["id"]) for q in selected}
    required_ids = {tid for qid in selected_ids for tid in haystack[qid]}
    trajectories = {
        str(row["id"]): row
        for row in trajectories_all
        if row.get("id") in required_ids
    }
    missing = required_ids - set(trajectories)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} trajectories")

    raw_documents = {tid: base.raw_memory(tr) for tid, tr in trajectories.items()}
    canonical_documents = {
        tid: base.unique_line_memory(tr) for tid, tr in trajectories.items()
    }
    methods = {
        "raw_sparse": base.SparseIndex(raw_documents),
        "unique_line_canonical": base.SparseIndex(canonical_documents),
    }
    raw_chars = sum(map(len, raw_documents.values()))
    canonical_chars = sum(map(len, canonical_documents.values()))
    storage = pd.DataFrame(
        [
            {"method": "raw_sparse", "stored_chars": raw_chars, "ratio_to_raw": 1.0},
            {
                "method": "unique_line_canonical",
                "stored_chars": canonical_chars,
                "ratio_to_raw": canonical_chars / raw_chars,
            },
        ]
    )
    storage.to_csv(output / "storage.csv", index=False)
    (output / "selected_question_ids.json").write_text(
        json.dumps([q["id"] for q in selected], indent=2)
    )

    # Retrieve all contexts before reader inference. This keeps reader calls paired and
    # prevents concurrent access to the sparse indexes from affecting timing or output.
    tasks: list[dict[str, Any]] = []
    for question_index, q in enumerate(selected, start=1):
        system_prompt = base.WEB_SYSTEM if q["domain"] == "web" else base.ENTERPRISE_SYSTEM
        for method, index in methods.items():
            context, retrieval_seconds = index.retrieve(
                str(q["question"]),
                list(haystack[q["id"]]),
                args.context_chars,
            )
            tasks.append(
                {
                    "question": q,
                    "question_index": question_index,
                    "method": method,
                    "system_prompt": system_prompt,
                    "context": context,
                    "retrieval_seconds": retrieval_seconds,
                }
            )

    checkpoint = output / "runs.csv"
    rows: list[dict[str, Any]] = []
    if checkpoint.exists():
        rows = pd.read_csv(checkpoint).to_dict("records")
    completed = {(str(r["question_id"]), str(r["method"])) for r in rows}
    pending = [
        task
        for task in tasks
        if (str(task["question"]["id"]), str(task["method"])) not in completed
    ]

    def run_one(task: dict[str, Any]) -> dict[str, Any]:
        q = task["question"]
        response, reader_seconds = base.call_reader(
            args.base_url,
            args.model,
            task["system_prompt"],
            task["context"],
            str(q["question"]),
            timeout=900,
        )
        parsed = extract_boxed_answer(response)
        score = False
        error = ""
        try:
            score = score_to_bool(
                eval_from_spec(q["eval_function"], parsed, q["answer"])
            )
            if is_unknown(parsed):
                score = False
        except Exception as exc:
            error = repr(exc)
        return {
            "question_id": q["id"],
            "question_index": task["question_index"],
            "domain": q["domain"],
            "question_type": q["question_type"],
            "eval_function": q["eval_function"],
            "method": task["method"],
            "correct": int(score),
            "response_raw": response,
            "response_parsed_boxed": parsed,
            "gold_answer": q["answer"],
            "context_chars": len(task["context"]),
            "retrieval_seconds": task["retrieval_seconds"],
            "reader_seconds": reader_seconds,
            "score_error": error,
        }

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(run_one, task): task for task in pending}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as exc:
                q = task["question"]
                row = {
                    "question_id": q["id"],
                    "question_index": task["question_index"],
                    "domain": q["domain"],
                    "question_type": q["question_type"],
                    "eval_function": q["eval_function"],
                    "method": task["method"],
                    "correct": 0,
                    "response_raw": "",
                    "response_parsed_boxed": "",
                    "gold_answer": q["answer"],
                    "context_chars": len(task["context"]),
                    "retrieval_seconds": task["retrieval_seconds"],
                    "reader_seconds": np.nan,
                    "score_error": repr(exc),
                }
            rows.append(row)
            pd.DataFrame(rows).sort_values(
                ["question_index", "method"]
            ).to_csv(checkpoint, index=False)
            print(
                json.dumps(
                    {
                        "done": len(rows),
                        "total": len(tasks),
                        "question_id": row["question_id"],
                        "method": row["method"],
                        "correct": row["correct"],
                        "reader_seconds": row["reader_seconds"],
                        "error": row["score_error"],
                    }
                ),
                flush=True,
            )

    runs = pd.DataFrame(rows).sort_values(["question_index", "method"])
    runs.to_csv(checkpoint, index=False)
    summary = (
        runs.groupby("method")
        .agg(
            questions=("question_id", "nunique"),
            accuracy=("correct", "mean"),
            median_reader_seconds=("reader_seconds", "median"),
            median_retrieval_ms=(
                "retrieval_seconds", lambda x: 1000 * float(np.median(x))
            ),
            mean_context_chars=("context_chars", "mean"),
            request_error_rate=("score_error", lambda x: float(np.mean(x.astype(str) != "")),
        )
        .reset_index()
        .merge(storage, on="method", how="left")
    )
    summary.to_csv(output / "summary.csv", index=False)

    pivot = runs.pivot(index="question_id", columns="method", values="correct").dropna()
    differences = (
        pivot["unique_line_canonical"] - pivot["raw_sparse"]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(20260830)
    boot = differences[
        rng.integers(0, len(differences), size=(20000, len(differences)))
    ].mean(axis=1)
    unique_only = int(
        ((pivot["raw_sparse"] == 0) & (pivot["unique_line_canonical"] == 1)).sum()
    )
    raw_only = int(
        ((pivot["raw_sparse"] == 1) & (pivot["unique_line_canonical"] == 0)).sum()
    )
    paired = {
        "questions": int(len(pivot)),
        "accuracy_gain_unique_minus_raw": float(differences.mean()),
        "bootstrap_95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_95_high": float(np.quantile(boot, 0.975)),
        "unique_only_correct": unique_only,
        "raw_only_correct": raw_only,
        "reader_model": args.model,
        "context_chars": args.context_chars,
        "selection_seed": args.selection_seed,
        "workers": args.workers,
        "elapsed_seconds": time.time() - started,
        "scope": (
            "Official LME-V2 small data, exact benchmark system/user prompt shape, "
            "exact deterministic evaluators; local Qwen3.5-0.8B substitute reader, "
            "not the official Qwen3.5-9B leaderboard reader."
        ),
    }
    (output / "paired_result.json").write_text(json.dumps(paired, indent=2))
    (
        runs.groupby(["method", "domain", "question_type"])
        .agg(questions=("question_id", "nunique"), accuracy=("correct", "mean"))
        .reset_index()
        .to_csv(output / "by_type.csv", index=False)
    )
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("PAIRED", json.dumps(paired, sort_keys=True))


if __name__ == "__main__":
    main()
