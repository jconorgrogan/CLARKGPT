#!/usr/bin/env python3
"""Patch the public LongMemEval-V2 reader for concise deterministic outputs.

The benchmark's deterministic evaluators require a short final answer. This
adapter asks the local llama.cpp reader for one strict JSON object and parses
its ``answer`` field before official scoring.
"""
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

    needle = "Answer based on your memory of the environment. "
    replacement = (
        "Answer based on your memory of the environment. "
        "Give the shortest supported final answer. "
    )
    text = replace_exact(
        text,
        needle,
        replacement,
        label="reader system-prompt sentence",
        expected=2,
    )

    old_question = '    question_block = f"\\n\\n### Question to answer:\\n{question_text}"\n'
    new_question = (
        "    question_block = (\n"
        '        f"\\n\\n### Question to answer:\\n{question_text}"\n'
        '        "\\n\\n### Required response:\\nReturn one JSON object with one field named answer. "\n'
        '        "Put only the final benchmark answer in that field."\n'
        "    )\n"
    )
    text = replace_exact(
        text,
        old_question,
        new_question,
        label="question_block construction",
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
                            "maxLength": 160,
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
    print(f"Patched reader harness: {args.harness}")


if __name__ == "__main__":
    main()
