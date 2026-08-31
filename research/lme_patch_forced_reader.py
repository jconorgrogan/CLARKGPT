#!/usr/bin/env python3
"""Patch LongMemEval-V2 for a deterministic forced-answer reader diagnostic."""
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
        'If you do not know the answer, output exactly \\boxed{UNKNOWN}. ',
        'Select the best-supported answer even when the evidence is incomplete. ',
        label="reader abstention instruction",
        expected=2,
    )
    text = replace_exact(
        text,
        'Do not guess. Never attempt to guess an answer if you are not sure. ',
        'Do not output UNKNOWN. Give the shortest final benchmark answer. ',
        label="reader guessing instruction",
        expected=2,
    )

    old_question = '    question_block = f"\\n\\n### Question to answer:\\n{question_text}"\n'
    new_question = (
        "    question_block = (\n"
        '        f"\\n\\n### Question to answer:\\n{question_text}"\n'
        '        "\\n\\n### Required response:\\nReturn one JSON object with one field named answer. "\n'
        '        "Put only the shortest final benchmark answer in that field. "\n'
        '        "For multiple-choice questions, put only the option letter."\n'
        "    )\n"
    )
    text = replace_exact(
        text,
        old_question,
        new_question,
        label="question block construction",
    )

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
                "name": "final_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 240,
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
'''
    text = replace_exact(
        text,
        old_request,
        new_request,
        label="reader request construction",
    )

    old_parser = '        parsed_answer = extract_boxed_answer(response_raw)\n'
    new_parser = '''        try:
            structured = json.loads(response_raw)
            answer_field = structured.get("answer", "") if isinstance(structured, dict) else structured
            parsed_answer = str(answer_field).strip()
            boxed_answer = extract_boxed_answer(parsed_answer)
            if boxed_answer:
                parsed_answer = boxed_answer
        except Exception:
            parsed_answer = extract_boxed_answer(response_raw)
'''
    text = replace_exact(
        text,
        old_parser,
        new_parser,
        label="reader response parser",
    )

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("harness", type=Path)
    args = parser.parse_args()
    patch_harness(args.harness)
    print(f"Patched forced-answer reader harness: {args.harness}")


if __name__ == "__main__":
    main()
