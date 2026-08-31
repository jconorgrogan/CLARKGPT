#!/usr/bin/env python3
"""Patch LongMemEval-V2 for a deterministic single-letter multiple-choice reader."""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_exact(text: str, old: str, new: str, *, label: str, expected: int = 1) -> str:
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"Expected {expected} occurrence(s) of {label}, found {count}")
    return text.replace(old, new)


def patch_harness(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_exact(
        text,
        'Answer based on your memory of the environment. ',
        'Answer using only the supplied memory context. ',
        label="reader memory instruction",
        expected=2,
    )
    text = replace_exact(
        text,
        r'If you do not know the answer, output exactly \\boxed{UNKNOWN}. ',
        'Select the option best supported by the memory context. ',
        label="reader abstention instruction",
        expected=2,
    )
    text = replace_exact(
        text,
        'Do not guess. Never attempt to guess an answer if you are not sure. ',
        'Choose one option. Return exactly one uppercase option letter. ',
        label="reader guessing instruction",
        expected=2,
    )

    old_question = '    question_block = f"\\n\\n### Question to answer:\\n{question_text}"\n'
    new_question = (
        "    question_block = (\n"
        '        f"\\n\\n### Question to answer:\\n{question_text}"\n'
        '        "\\n\\n### Required response:\\nChoose the best-supported option. "\n'
        '        "Return one JSON object with one field named answer containing exactly one uppercase letter A-H."\n'
        "    )\n"
    )
    text = replace_exact(text, old_question, new_question, label="question block construction")

    old_request = '''    req: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "timeout": args.timeout_seconds,
    }
'''
    new_request = '''    req: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "timeout": args.timeout_seconds,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "multiple_choice_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "E", "F", "G", "H"],
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
'''
    text = replace_exact(text, old_request, new_request, label="reader request construction")

    old_parser = '        parsed_answer = extract_boxed_answer(response_raw)\n'
    new_parser = '''        try:
            structured = json.loads(response_raw)
            answer_field = structured.get("answer", "") if isinstance(structured, dict) else structured
            parsed_answer = str(answer_field).strip().upper()
        except Exception:
            parsed_answer = extract_boxed_answer(response_raw).strip().upper()
'''
    text = replace_exact(text, old_parser, new_parser, label="reader response parser")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("harness", type=Path)
    args = parser.parse_args()
    patch_harness(args.harness)
    print(f"Patched grammar-constrained multiple-choice reader: {args.harness}")


if __name__ == "__main__":
    main()

# Frozen 30-question confirmation trigger; evaluation code is unchanged.
