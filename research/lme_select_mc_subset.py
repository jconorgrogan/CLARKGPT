from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

OPTION_RE = re.compile(r"(?m)^\s*([A-H])\.\s+\S")
VALID_ANSWERS = set("ABCDEFGH")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-domain", type=int, default=15)
    parser.add_argument("--seed", type=int, default=260832)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(data_root / "questions.jsonl")
    haystacks = json.loads((data_root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    manifest: dict[str, Any] = {
        "experiment": "anchored-mc-30-v1",
        "seed": args.seed,
        "selection_rule": "text-only deterministic questions with >=3 lettered options and single-letter gold format",
        "domains": {},
    }
    for domain in ("web", "enterprise"):
        eligible: list[dict[str, Any]] = []
        for row in questions:
            text = row.get("question")
            answer = str(row.get("answer", "")).strip().upper()
            options = sorted(set(OPTION_RE.findall(text or "")))
            if (
                row.get("domain") == domain
                and row.get("image") is None
                and not str(row.get("eval_function", "")).startswith("llm_")
                and isinstance(text, str)
                and len(options) >= 3
                and answer in VALID_ANSWERS
                and row.get("id") in haystacks
            ):
                eligible.append(row)

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
                f"Only selected {len(selected)} eligible multiple-choice {domain} questions; "
                f"eligible_by_type={ {name: len(values) for name, values in groups.items()} }"
            )

        (output_root / f"{domain}_questions.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        (output_root / f"{domain}_haystack.json").write_text(
            json.dumps({str(row["id"]): haystacks[str(row["id"])] for row in selected}, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["domains"][domain] = {
            "eligible_count": len(eligible),
            "selected_count": len(selected),
            "question_ids": [row["id"] for row in selected],
            "question_types": [row["question_type"] for row in selected],
            "gold_letters_sha_input": [str(row["answer"]).strip().upper() for row in selected],
            "haystack_sizes": [len(haystacks[str(row["id"])]) for row in selected],
        }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
