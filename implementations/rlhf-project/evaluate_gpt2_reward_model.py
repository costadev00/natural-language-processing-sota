#!/usr/bin/env python3
"""Evaluate a GPT-2 reward model on Dolly synthetic preference pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 reward model pairwise preferences.")
    parser.add_argument("--dataset-dir", default="rlhf_dolly_datasets/rm_synthetic")
    parser.add_argument("--reward-model", default="reward_gpt2/model")
    parser.add_argument("--output-dir", default="reward_gpt2/evaluation")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-category", type=int, default=3)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 10**9
    return tokenizer


def encode_prompt_response(prompt: str, response: str, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(f"{prompt}\n\n", add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(f"{response}{eos}", add_special_tokens=False)["input_ids"]
    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
    else:
        prompt_budget = max_length - len(response_ids)
        input_ids = prompt_ids[-prompt_budget:] + response_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def copy_rate(text: str, prompt: str) -> float:
    text_tokens = words(text)
    if not text_tokens:
        return 0.0
    prompt_tokens = set(words(prompt))
    return sum(token in prompt_tokens for token in text_tokens) / len(text_tokens)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def subset_test(dataset_dict: DatasetDict, max_eval_samples: int | None, seed: int) -> Dataset:
    dataset = dataset_dict["test"]
    if max_eval_samples is not None and max_eval_samples < len(dataset):
        return dataset.shuffle(seed=seed).select(range(max_eval_samples))
    return dataset


def score_texts(
    prompts: list[str],
    responses: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
) -> list[float]:
    encoded_rows = [encode_prompt_response(prompt, response, tokenizer, max_length) for prompt, response in zip(prompts, responses)]
    padded = tokenizer.pad(encoded_rows, padding=True, return_tensors="pt")
    padded = {key: value.to(device) for key, value in padded.items()}
    with torch.no_grad():
        logits = model(**padded).logits.squeeze(-1)
    return [float(value) for value in logits.detach().cpu().tolist()]


def evaluate_rows(dataset: Dataset, tokenizer: Any, model: Any, device: torch.device, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        prompts = list(batch["prompt"])
        chosen_responses = list(batch["chosen"])
        rejected_responses = list(batch["rejected"])
        chosen_rewards = score_texts(prompts, chosen_responses, tokenizer, model, device, args.max_length)
        rejected_rewards = score_texts(prompts, rejected_responses, tokenizer, model, device, args.max_length)

        for idx, chosen_reward in enumerate(chosen_rewards):
            rejected_reward = rejected_rewards[idx]
            margin = chosen_reward - rejected_reward
            rows.append(
                {
                    "category": str(batch["category"][idx]),
                    "source_id": str(batch["source_id"][idx]),
                    "prompt": str(batch["prompt"][idx]),
                    "chosen": str(batch["chosen"][idx]),
                    "rejected": str(batch["rejected"][idx]),
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "margin": margin,
                    "correct": margin > 0,
                    "pair_loss": float(-F.logsigmoid(torch.tensor(margin)).item()),
                    "chosen_words": len(words(str(batch["chosen"][idx]))),
                    "rejected_words": len(words(str(batch["rejected"][idx]))),
                    "chosen_copy_rate": copy_rate(str(batch["chosen"][idx]), str(batch["prompt"][idx])),
                    "rejected_copy_rate": copy_rate(str(batch["rejected"][idx]), str(batch["prompt"][idx])),
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "pairwise_accuracy": 0.0,
            "pair_loss": 0.0,
            "mean_margin": 0.0,
            "mean_chosen_reward": 0.0,
            "mean_rejected_reward": 0.0,
        }

    margins = [float(row["margin"]) for row in rows]
    losses = [float(row["pair_loss"]) for row in rows]
    chosen_rewards = [float(row["chosen_reward"]) for row in rows]
    rejected_rewards = [float(row["rejected_reward"]) for row in rows]
    accuracy = sum(bool(row["correct"]) for row in rows) / len(rows)

    return {
        "samples": len(rows),
        "pairwise_accuracy": float(accuracy),
        "pair_loss": float(sum(losses) / len(losses)),
        "mean_margin": float(sum(margins) / len(margins)),
        "median_margin": float(sorted(margins)[len(margins) // 2]),
        "mean_chosen_reward": float(sum(chosen_rewards) / len(chosen_rewards)),
        "mean_rejected_reward": float(sum(rejected_rewards) / len(rejected_rewards)),
        "chosen_reward_std": float(torch.tensor(chosen_rewards).std(unbiased=False).item()),
        "rejected_reward_std": float(torch.tensor(rejected_rewards).std(unbiased=False).item()),
    }


def summarize_by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    return {category: summarize_rows(category_rows) for category, category_rows in sorted(grouped.items())}


def summarize_correlations(rows: list[dict[str, Any]]) -> dict[str, float]:
    margins = [float(row["margin"]) for row in rows]
    return {
        "margin_vs_chosen_words": float(pearson(margins, [float(row["chosen_words"]) for row in rows])),
        "margin_vs_rejected_words": float(pearson(margins, [float(row["rejected_words"]) for row in rows])),
        "margin_vs_chosen_copy_rate": float(pearson(margins, [float(row["chosen_copy_rate"]) for row in rows])),
        "margin_vs_rejected_copy_rate": float(pearson(margins, [float(row["rejected_copy_rate"]) for row in rows])),
    }


def select_examples(rows: list[dict[str, Any]], samples_per_category: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    rng = random.Random(seed)
    examples = []
    for category in sorted(grouped):
        correct = [row for row in grouped[category] if row["correct"]]
        incorrect = [row for row in grouped[category] if not row["correct"]]
        rng.shuffle(correct)
        rng.shuffle(incorrect)
        examples.extend(incorrect[: max(1, samples_per_category // 2)])
        examples.extend(correct[:samples_per_category])
    return examples


def truncate(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, evaluation: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    summary = evaluation["summary"]
    lines = [
        "# GPT-2 Reward Model Evaluation",
        "",
        "## Summary",
        "",
        "- This reward model was trained on synthetic proxy preferences, not human preference labels.",
        "- Dolly reference responses are treated as chosen; sampled SFT GPT-2 responses are treated as rejected.",
        f"- Test pairwise accuracy: `{summary['pairwise_accuracy']:.4f}` over `{summary['samples']}` pairs.",
        f"- Test pair loss: `{summary['pair_loss']:.4f}`.",
        f"- Mean reward margin chosen minus rejected: `{summary['mean_margin']:.4f}`.",
        f"- Mean chosen reward: `{summary['mean_chosen_reward']:.4f}`; mean rejected reward: `{summary['mean_rejected_reward']:.4f}`.",
        "",
        "## Bias Checks",
        "",
        "| Signal | Pearson r with margin |",
        "| --- | ---: |",
    ]
    for key, value in evaluation["correlations"].items():
        lines.append(f"| {key} | {value:.4f} |")

    lines.extend(
        [
            "",
            "## Category Metrics",
            "",
            "| Category | Samples | Accuracy | Pair loss | Mean margin |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, metrics in evaluation["by_category"].items():
        lines.append(
            f"| {category} | {metrics['samples']} | {metrics['pairwise_accuracy']:.4f} | "
            f"{metrics['pair_loss']:.4f} | {metrics['mean_margin']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Qualitative Notes",
            "",
            "- Accuracy above 0.5 means the model learned to separate Dolly references from SFT samples in this proxy setup.",
            "- Very high margins would not prove human alignment; they could also mean the RM learned artifacts of generated rejected responses.",
            "- This RM is useful for exercising the RLHF pipeline, but a production RM needs human or otherwise audited preference pairs.",
            "",
            "## Examples",
            "",
        ]
    )
    for row in examples[: min(12, len(examples))]:
        lines.extend(
            [
                f"### {row['category']} / {row['source_id']}",
                "",
                f"**Correct:** `{row['correct']}`; **margin:** `{row['margin']:.4f}`; "
                f"**chosen reward:** `{row['chosen_reward']:.4f}`; **rejected reward:** `{row['rejected_reward']:.4f}`.",
                "",
                f"**Prompt:** {truncate(row['prompt'])}",
                "",
                f"**Chosen/Dolly:** {truncate(row['chosen'])}",
                "",
                f"**Rejected/SFT sample:** {truncate(row['rejected'])}",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = load_from_disk(args.dataset_dir)
    dataset = subset_test(dataset_dict, args.max_eval_samples, args.seed)

    tokenizer = prepare_tokenizer(args.reward_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rows = evaluate_rows(dataset, tokenizer, model, device, args)
    examples = select_examples(rows, args.samples_per_category, args.seed)
    evaluation = {
        "reward_model": args.reward_model,
        "dataset_dir": args.dataset_dir,
        "warning": "Synthetic proxy preferences, not human preference labels.",
        "summary": summarize_rows(rows),
        "by_category": summarize_by_category(rows),
        "correlations": summarize_correlations(rows),
    }

    write_json(output_dir / "evaluation.json", evaluation)
    write_jsonl(output_dir / "examples.jsonl", examples)
    write_report(output_dir / "analysis_report.md", evaluation, examples)

    print(f"Saved evaluation metrics to {output_dir / 'evaluation.json'}")
    print(f"Saved examples to {output_dir / 'examples.jsonl'}")
    print(f"Saved report to {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
