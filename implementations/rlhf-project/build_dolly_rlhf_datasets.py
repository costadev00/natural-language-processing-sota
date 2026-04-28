#!/usr/bin/env python3
"""Build Dolly 15k SFT, RM-schema, and PPO datasets in InstructGPT style."""

from __future__ import annotations

import argparse
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASET = "databricks/databricks-dolly-15k"
EXPECTED_TOTAL_ROWS = 15011
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local Hugging Face DatasetDict artifacts for RLHF stages from Dolly 15k."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--output", default="rlhf_dolly_datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output subdirectories before saving new datasets.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0.")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def render_prompt(instruction: str, context: str) -> str:
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(count) for count in raw_counts]
    remainder = total - sum(counts)

    ranked_remainders = sorted(
        range(len(ratios)),
        key=lambda idx: (raw_counts[idx] - counts[idx], ratios[idx]),
        reverse=True,
    )
    for idx in ranked_remainders[:remainder]:
        counts[idx] += 1

    return counts[0], counts[1], counts[2]


def load_normalized_records(dataset_name: str) -> list[dict[str, object]]:
    raw = load_dataset(dataset_name, split="train")
    records: list[dict[str, object]] = []

    for idx, row in enumerate(raw):
        instruction = clean_text(row["instruction"])
        context = clean_text(row["context"])
        response = clean_text(row["response"])
        category = clean_text(row["category"])
        prompt = render_prompt(instruction, context)

        if not instruction:
            raise ValueError(f"Empty instruction at source row {idx}.")
        if not response:
            raise ValueError(f"Empty response at source row {idx}.")
        if not category:
            raise ValueError(f"Empty category at source row {idx}.")
        if not prompt:
            raise ValueError(f"Empty prompt at source row {idx}.")

        records.append(
            {
                "instruction": instruction,
                "context": context,
                "response": response,
                "category": category,
                "prompt": prompt,
                "source_id": f"{dataset_name}:train:{idx}",
                "has_context": bool(context),
            }
        )

    return records


def stratified_split(
    records: Iterable[dict[str, object]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["category"])].append(record)

    rng = random.Random(seed)
    split_records: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}

    for category in sorted(grouped):
        category_records = list(grouped[category])
        rng.shuffle(category_records)
        train_count, validation_count, test_count = allocate_counts(len(category_records), ratios)

        train_end = train_count
        validation_end = train_end + validation_count

        split_records["train"].extend(category_records[:train_end])
        split_records["validation"].extend(category_records[train_end:validation_end])
        split_records["test"].extend(category_records[validation_end : validation_end + test_count])

    for split in SPLITS:
        rng.shuffle(split_records[split])

    return split_records


def to_sft_rows(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        prompt = str(record["prompt"])
        completion = str(record["response"])
        rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "text": f"{prompt}\n\n{completion}",
                "instruction": str(record["instruction"]),
                "context": str(record["context"]),
                "category": str(record["category"]),
                "source_id": str(record["source_id"]),
                "has_context": bool(record["has_context"]),
            }
        )
    return rows


def to_rm_schema_rows(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        rows.append(
            {
                "prompt": str(record["prompt"]),
                "chosen": "",
                "rejected": "",
                "reference_response": str(record["response"]),
                "category": str(record["category"]),
                "source_id": str(record["source_id"]),
                "has_context": bool(record["has_context"]),
                "ready_for_rm": False,
            }
        )
    return rows


def to_ppo_rows(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        rows.append(
            {
                "prompt": str(record["prompt"]),
                "category": str(record["category"]),
                "source_id": str(record["source_id"]),
                "has_context": bool(record["has_context"]),
            }
        )
    return rows


def make_dataset_dict(split_records: dict[str, list[dict[str, object]]], kind: str) -> DatasetDict:
    converters = {
        "sft": to_sft_rows,
        "rm_schema": to_rm_schema_rows,
        "ppo": to_ppo_rows,
    }
    converter = converters[kind]
    return DatasetDict({split: Dataset.from_list(converter(split_records[split])) for split in SPLITS})


def validate_dataset_dicts(outputs: dict[str, DatasetDict], expected_total: int) -> None:
    expected_columns = {
        "sft": [
            "prompt",
            "completion",
            "text",
            "instruction",
            "context",
            "category",
            "source_id",
            "has_context",
        ],
        "rm_schema": [
            "prompt",
            "chosen",
            "rejected",
            "reference_response",
            "category",
            "source_id",
            "has_context",
            "ready_for_rm",
        ],
        "ppo": ["prompt", "category", "source_id", "has_context"],
    }

    for kind, dataset_dict in outputs.items():
        if tuple(dataset_dict.keys()) != SPLITS:
            raise ValueError(f"{kind} splits are {tuple(dataset_dict.keys())}, expected {SPLITS}.")

        total = sum(len(dataset_dict[split]) for split in SPLITS)
        if total != expected_total:
            raise ValueError(f"{kind} has {total} rows, expected {expected_total}.")

        seen_source_ids: set[str] = set()
        for split in SPLITS:
            dataset = dataset_dict[split]
            if dataset.column_names != expected_columns[kind]:
                raise ValueError(f"{kind}/{split} columns are {dataset.column_names}.")
            if any(not prompt for prompt in dataset["prompt"]):
                raise ValueError(f"{kind}/{split} contains empty prompts.")

            split_source_ids = set(dataset["source_id"])
            if len(split_source_ids) != len(dataset):
                raise ValueError(f"{kind}/{split} contains duplicate source_id values.")
            if seen_source_ids.intersection(split_source_ids):
                raise ValueError(f"{kind} source_id overlap across splits.")
            seen_source_ids.update(split_source_ids)

            if kind == "sft":
                if any(not text for text in dataset["text"]):
                    raise ValueError(f"{kind}/{split} contains empty text values.")
            elif kind == "rm_schema":
                if any(dataset["chosen"]) or any(dataset["rejected"]):
                    raise ValueError(f"{kind}/{split} contains non-empty chosen/rejected fields.")
                if any(dataset["ready_for_rm"]):
                    raise ValueError(f"{kind}/{split} contains ready_for_rm=True values.")
            elif kind == "ppo":
                forbidden = {"completion", "response", "text", "chosen", "rejected"}
                if forbidden.intersection(dataset.column_names):
                    raise ValueError(f"{kind}/{split} contains forbidden response columns.")


def save_outputs(outputs: dict[str, DatasetDict], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, dataset_dict in outputs.items():
        path = output_dir / kind
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
            shutil.rmtree(path)
        dataset_dict.save_to_disk(str(path))


def print_summary(outputs: dict[str, DatasetDict]) -> None:
    for kind, dataset_dict in outputs.items():
        print(f"{kind}:")
        for split in SPLITS:
            dataset = dataset_dict[split]
            categories = Counter(dataset["category"])
            category_summary = ", ".join(f"{key}={categories[key]}" for key in sorted(categories))
            print(f"  {split}: {len(dataset)} rows ({category_summary})")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.validation_ratio, args.test_ratio)

    records = load_normalized_records(args.dataset_name)
    expected_total = EXPECTED_TOTAL_ROWS if args.dataset_name == DEFAULT_DATASET else len(records)
    if len(records) != expected_total:
        raise ValueError(f"Loaded {len(records)} rows, expected {expected_total}.")

    split_records = stratified_split(
        records,
        (args.train_ratio, args.validation_ratio, args.test_ratio),
        args.seed,
    )

    outputs = {
        "sft": make_dataset_dict(split_records, "sft"),
        "rm_schema": make_dataset_dict(split_records, "rm_schema"),
        "ppo": make_dataset_dict(split_records, "ppo"),
    }
    validate_dataset_dicts(outputs, expected_total)
    save_outputs(outputs, Path(args.output), args.overwrite)
    print_summary(outputs)


if __name__ == "__main__":
    main()
