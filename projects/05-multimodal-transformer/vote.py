#!/usr/bin/env python3
"""Collect blinded human preferences for a response JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, help="Path to responses.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to votes.csv beside responses.jsonl")
    parser.add_argument("--seed", type=int, default=5029)
    parser.add_argument("--resume", action="store_true", help="Skip prompts already present in the votes file.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_votes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["prompt_id"] for row in csv.DictReader(handle)}


def print_block(title: str, text: str) -> None:
    bar = "=" * 88
    print(f"\n{bar}\n{title}\n{bar}")
    print(text.strip() or "[empty response]")


def write_header_if_needed(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prompt_id",
                "category",
                "left_model",
                "right_model",
                "left_provider",
                "right_provider",
                "vote",
                "winner_model",
                "notes",
            ],
        )
        writer.writeheader()


def append_vote(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    output = args.output or (args.responses.parent / "votes.csv")
    rows = read_jsonl(args.responses)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)

    completed = existing_votes(output) if args.resume else set()
    write_header_if_needed(output)

    rng = random.Random(args.seed)
    prompt_ids = sorted(by_prompt)
    for ordinal, prompt_id in enumerate(prompt_ids, start=1):
        if prompt_id in completed:
            continue
        candidates = by_prompt[prompt_id]
        if len(candidates) < 2:
            print(f"Skipping {prompt_id}: only {len(candidates)} response(s).")
            continue

        pair = candidates[:2]
        rng.shuffle(pair)
        left, right = pair

        print("\n\n" + "#" * 100)
        print(f"Prompt {ordinal}/{len(prompt_ids)}: {prompt_id} ({left.get('category')})")
        print("#" * 100)
        print(left.get("prompt", "").strip())
        if left.get("images"):
            print("\nImages:")
            for image in left["images"]:
                print(f"- {image.get('id')}: {image.get('path')}")

        print_block("Response A", left.get("response_text") or left.get("error") or "")
        print_block("Response B", right.get("response_text") or right.get("error") or "")

        while True:
            choice = input("\nVote [a/b/t=tie/s=skip/q=quit]: ").strip().lower()
            if choice in {"a", "b", "t", "s", "q"}:
                break
        if choice == "q":
            print(f"Votes saved in {output}")
            return 0
        if choice == "s":
            continue

        notes = input("Optional notes: ").strip()
        winner_model = ""
        if choice == "a":
            winner_model = left["model_id"]
        elif choice == "b":
            winner_model = right["model_id"]
        else:
            winner_model = "tie"

        append_vote(
            output,
            {
                "prompt_id": prompt_id,
                "category": left.get("category", ""),
                "left_model": left.get("model_id", ""),
                "right_model": right.get("model_id", ""),
                "left_provider": left.get("provider", ""),
                "right_provider": right.get("provider", ""),
                "vote": choice,
                "winner_model": winner_model,
                "notes": notes,
            },
        )

    print(f"Votes saved in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
