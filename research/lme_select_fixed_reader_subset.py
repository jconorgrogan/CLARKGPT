from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-domain", type=int, default=12)
    parser.add_argument("--seed", type=int, default=260830)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    questions = read_jsonl(data_root / "questions.jsonl")
    haystacks = json.loads(
        (data_root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8")
    )
    rng = random.Random(args.seed)

    manifest: dict[str, Any] = {"seed": args.seed, "domains": {}}
    for domain in ("web", "enterprise"):
        eligible = [
            row
            for row in questions
            if row.get("domain") == domain
            and row.get("image") is None
            and not str(row.get("eval_function", "")).startswith("llm_")
            and isinstance(row.get("question"), str)
            and row.get("id") in haystacks
        ]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            groups[str(row.get("question_type"))].append(row)
        for values in groups.values():
            rng.shuffle(values)

        selected: list[dict[str, Any]] = []
        group_names = sorted(groups)
        while len(selected) < args.per_domain:
            progressed = False
            for name in group_names:
                if groups[name] and len(selected) < args.per_domain:
                    selected.append(groups[name].pop())
                    progressed = True
            if not progressed:
                break
        if len(selected) < args.per_domain:
            raise RuntimeError(
                f"Only selected {len(selected)} eligible {domain} questions"
            )

        question_path = output_root / f"{domain}_questions.json"
        haystack_path = output_root / f"{domain}_haystack.json"
        question_path.write_text(
            json.dumps(selected, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        haystack_path.write_text(
            json.dumps(
                {str(row["id"]): haystacks[str(row["id"])] for row in selected},
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["domains"][domain] = {
            "count": len(selected),
            "question_ids": [row["id"] for row in selected],
            "question_types": [row["question_type"] for row in selected],
            "eval_functions": [row["eval_function"] for row in selected],
            "haystack_sizes": [len(haystacks[str(row["id"])]) for row in selected],
        }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
