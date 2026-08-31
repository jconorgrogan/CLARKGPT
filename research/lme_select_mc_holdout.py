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
DEVELOPMENT_IDS = {
    "web": {
        "06e965cf", "07fff330", "1defc293", "1e011eb5", "23aecb38",
        "30fab98c", "674a6972", "91d0775e", "b828a6b2", "be541aa6",
        "c39f6edf", "c56e69ca", "c738b934", "c7e367e2", "dae9f7e9",
    },
    "enterprise": {
        "100ff132", "233f9f09", "4dffe641", "5edd2533", "78686f4e",
        "7e32e4a2", "7ea13f14", "9e05978c", "b8cabd09", "bdb825a3",
        "bfb3bcc4", "d63e8a7f", "e033e796", "e334d5c6", "f9c65fe6",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eligible_rows(questions: list[dict[str, Any]], haystacks: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            and str(row.get("id")) in haystacks
        ):
            rows.append(row)
    return rows


def balanced_select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("question_type"))].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    group_names = sorted(groups)
    while len(selected) < count:
        progressed = False
        for name in group_names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(f"Requested {count}, selected {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--per-domain", type=int, default=15)
    parser.add_argument("--holdout-seed", type=int, default=260833)
    args = parser.parse_args()

    root = Path(args.data_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(root / "questions.jsonl")
    haystacks = json.loads((root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8"))

    manifest: dict[str, Any] = {
        "experiment": "incidence-heldout-mc-30-v1",
        "development_experiment": "anchored-mc-30-v1",
        "holdout_seed": args.holdout_seed,
        "selection_rule": "balanced text-only deterministic MC; exact exclusion of all 30 prior development IDs",
        "domains": {},
    }
    for offset, domain in enumerate(("web", "enterprise")):
        eligible = eligible_rows(questions, haystacks, domain)
        eligible_ids = {str(row["id"]) for row in eligible}
        excluded = DEVELOPMENT_IDS[domain]
        missing_excluded = excluded - eligible_ids
        if missing_excluded:
            raise RuntimeError(f"Recorded development IDs missing from eligible {domain} pool: {sorted(missing_excluded)}")
        remaining = [row for row in eligible if str(row["id"]) not in excluded]
        heldout = balanced_select(remaining, args.per_domain, args.holdout_seed + offset)
        heldout_ids = {str(row["id"]) for row in heldout}
        overlap = excluded & heldout_ids
        if overlap:
            raise RuntimeError(f"Development/holdout overlap: {sorted(overlap)}")

        (out / f"{domain}_questions.json").write_text(json.dumps(heldout, indent=2) + "\n")
        (out / f"{domain}_haystack.json").write_text(
            json.dumps({str(row["id"]): haystacks[str(row["id"])] for row in heldout}, indent=2) + "\n"
        )
        manifest["domains"][domain] = {
            "eligible_count": len(eligible),
            "development_ids": sorted(excluded),
            "heldout_ids": [str(row["id"]) for row in heldout],
            "heldout_types": [str(row.get("question_type")) for row in heldout],
            "haystack_sizes": [len(haystacks[str(row["id"])]) for row in heldout],
            "overlap_count": 0,
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
