#!/usr/bin/env python3
"""Extract compact audited rows from a LongMemEval-V2 harness run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--completion-cap", type=int, default=192)
    args = parser.parse_args()

    per_question = args.run_dir / "per_question.jsonl"
    if not per_question.exists():
        raise SystemExit(f"Missing official per-question output: {per_question}")

    records = [
        json.loads(line)
        for line in per_question.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit("Official harness produced zero per-question records")

    rows: list[dict[str, object]] = []
    for record in records:
        metadata = record.get("memory_post_query_metadata") or {}
        usage = record.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        rows.append(
            {
                "method": args.method,
                "domain": args.domain,
                "question_id": record["question_id"],
                "question_type": record["question_type"],
                "score": float(record["score"]),
                "memory_query_seconds": float(record["memory_query_duration_seconds"]),
                "memory_context_tokens": int(record["memory_context_token_count"]),
                "raw_characters": metadata.get("raw_characters"),
                "stored_characters": metadata.get("stored_characters"),
                "storage_ratio": metadata.get("storage_ratio"),
                "completion_tokens": completion_tokens,
                "hit_completion_cap": completion_tokens >= args.completion_cap,
                "response": record.get("response_parsed_boxed", ""),
                "response_raw": record.get("response_raw", ""),
                "gold": record.get("answer_gold", ""),
            }
        )

    frame = pd.DataFrame(rows)
    expected = frame["question_id"].nunique()
    if expected != len(frame):
        raise SystemExit("Duplicate question IDs in official harness output")
    output = args.run_dir / "compact.csv"
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
