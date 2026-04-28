#!/usr/bin/env python3
"""Evaluate the GPT-2 SFT model against the base GPT-2 model."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 SFT outputs and test perplexity.")
    parser.add_argument("--dataset-dir", default="rlhf_dolly_datasets/sft")
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--sft-model", default="sft_gpt2/model")
    parser.add_argument("--output-dir", default="sft_gpt2/evaluation")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--samples-per-category", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    return parser.parse_args()


@dataclass
class DataCollatorForCompletionOnlyLM:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [label + [-100] * (max_length - len(label)) for label in labels],
            dtype=torch.long,
        )
        return batch


def tokenize_examples(examples: dict[str, list[str]], tokenizer: Any, max_length: int) -> dict[str, list[Any]]:
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    eos = tokenizer.eos_token or ""

    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        prompt_prefix = f"{prompt}\n\n"
        full_text = f"{prompt_prefix}{completion}{eos}"
        tokenized = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
        prompt_ids = tokenizer(
            prompt_prefix,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        labels = list(tokenized["input_ids"])
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        input_ids_batch.append(tokenized["input_ids"])
        attention_mask_batch.append(tokenized["attention_mask"])
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
    }


def has_supervised_tokens(example: dict[str, Any]) -> bool:
    return any(label != -100 for label in example["labels"])


def supports_training_arg(name: str) -> bool:
    return name in inspect.signature(TrainingArguments.__init__).parameters


def supported_training_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def trainer_tokenizer_kwarg(tokenizer: Any) -> dict[str, Any]:
    supported = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in supported:
        return {"processing_class": tokenizer}
    if "tokenizer" in supported:
        return {"tokenizer": tokenizer}
    return {}


def make_eval_args(output_dir: Path, args: argparse.Namespace, suffix: str) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / f"tmp_{suffix}"),
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "report_to": "none",
        "seed": args.seed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
    }
    if supports_training_arg("bf16"):
        kwargs["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if supports_training_arg("fp16"):
        kwargs["fp16"] = bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
    return TrainingArguments(**supported_training_kwargs(kwargs))


def prepare_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prepare_eval_dataset(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    max_length: int,
    max_eval_samples: int | None,
    seed: int,
) -> Dataset:
    dataset = dataset_dict["test"]
    if max_eval_samples is not None and max_eval_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_eval_samples))
    return dataset.map(
        lambda batch: tokenize_examples(batch, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing test",
    ).filter(has_supervised_tokens, desc="Filtering test")


def evaluate_loss(
    model_name_or_path: str,
    dataset_dict: DatasetDict,
    output_dir: Path,
    args: argparse.Namespace,
    suffix: str,
) -> dict[str, float]:
    tokenizer = prepare_tokenizer(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    eval_dataset = prepare_eval_dataset(dataset_dict, tokenizer, args.max_length, args.max_eval_samples, args.seed)
    trainer = Trainer(
        model=model,
        args=make_eval_args(output_dir, args, suffix),
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCompletionOnlyLM(tokenizer),
        **trainer_tokenizer_kwarg(tokenizer),
    )
    metrics = trainer.evaluate()
    loss = float(metrics["eval_loss"])
    return {
        "loss": loss,
        "perplexity": float(math.exp(loss)),
        "samples": len(eval_dataset),
    }


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = words(prediction)
    ref_tokens = words(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    prev = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        curr = [0]
        for idx, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[idx]))
        prev = curr

    lcs = prev[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def copy_rate(prediction: str, prompt: str) -> float:
    pred_tokens = words(prediction)
    if not pred_tokens:
        return 0.0
    prompt_tokens = set(words(prompt))
    return sum(token in prompt_tokens for token in pred_tokens) / len(pred_tokens)


def select_generation_examples(dataset_dict: DatasetDict, samples_per_category: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_dict["test"]:
        grouped[row["category"]].append(row)

    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    for category in sorted(grouped):
        rows = grouped[category]
        rng.shuffle(rows)
        examples.extend(rows[:samples_per_category])
    return examples


def load_generation_model(model_name_or_path: str) -> tuple[Any, Any, torch.device]:
    tokenizer = prepare_tokenizer(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
) -> str:
    max_input_tokens = max(16, max_length - max_new_tokens)
    generation_prompt = f"{prompt}\n\n"
    encoded = tokenizer(
        generation_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def summarize_generation_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    rouge_values = [row[f"{prefix}_rouge_l_f1"] for row in rows]
    copy_values = [row[f"{prefix}_copy_rate"] for row in rows]
    lengths = [len(words(row[f"{prefix}_generation"])) for row in rows]
    return {
        "rouge_l_f1_mean": float(sum(rouge_values) / len(rouge_values)) if rouge_values else 0.0,
        "copy_rate_mean": float(sum(copy_values) / len(copy_values)) if copy_values else 0.0,
        "response_words_mean": float(sum(lengths) / len(lengths)) if lengths else 0.0,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_report(path: Path, evaluation: dict[str, Any], generations: list[dict[str, Any]]) -> None:
    delta_loss = evaluation["base"]["loss"] - evaluation["sft"]["loss"]
    delta_ppl = evaluation["base"]["perplexity"] - evaluation["sft"]["perplexity"]
    lines = [
        "# GPT-2 SFT Evaluation",
        "",
        "## Summary",
        "",
        f"- Base GPT-2 test loss: `{evaluation['base']['loss']:.4f}`; perplexity: `{evaluation['base']['perplexity']:.2f}`.",
        f"- SFT GPT-2 test loss: `{evaluation['sft']['loss']:.4f}`; perplexity: `{evaluation['sft']['perplexity']:.2f}`.",
        f"- Loss improvement: `{delta_loss:.4f}`; perplexity improvement: `{delta_ppl:.2f}`.",
        f"- Generation sample size: `{len(generations)}` examples.",
        "",
        "## Lightweight Generation Metrics",
        "",
        "| Model | ROUGE-L F1 mean | Prompt copy rate | Mean response words |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| GPT-2 base | {evaluation['base_generation']['rouge_l_f1_mean']:.4f} | "
            f"{evaluation['base_generation']['copy_rate_mean']:.4f} | "
            f"{evaluation['base_generation']['response_words_mean']:.1f} |"
        ),
        (
            f"| GPT-2 SFT | {evaluation['sft_generation']['rouge_l_f1_mean']:.4f} | "
            f"{evaluation['sft_generation']['copy_rate_mean']:.4f} | "
            f"{evaluation['sft_generation']['response_words_mean']:.1f} |"
        ),
        "",
        "## Qualitative Notes",
        "",
        "- The loss/perplexity comparison is the strongest automatic signal here because labels are masked to score only the answer tokens.",
        "- ROUGE-L is useful for rough overlap with Dolly references, but it under-rates valid paraphrases and creative answers.",
        "- Prompt copy rate is approximate; high values can be normal for extractive or closed-domain examples.",
        "",
        "## Examples",
        "",
    ]

    for row in generations[: min(8, len(generations))]:
        lines.extend(
            [
                f"### {row['category']} / {row['source_id']}",
                "",
                f"**Prompt:** {truncate(row['prompt'])}",
                "",
                f"**Reference:** {truncate(row['reference'])}",
                "",
                f"**GPT-2 base:** {truncate(row['base_generation'])}",
                "",
                f"**GPT-2 SFT:** {truncate(row['sft_generation'])}",
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

    base_loss = evaluate_loss(args.base_model, dataset_dict, output_dir, args, "base")
    sft_loss = evaluate_loss(args.sft_model, dataset_dict, output_dir, args, "sft")

    examples = select_generation_examples(dataset_dict, args.samples_per_category, args.seed)
    base_tokenizer, base_model, base_device = load_generation_model(args.base_model)
    sft_tokenizer, sft_model, sft_device = load_generation_model(args.sft_model)

    generation_rows = []
    for row in examples:
        base_generation = generate_response(
            row["prompt"],
            base_tokenizer,
            base_model,
            base_device,
            args.max_length,
            args.max_new_tokens,
        )
        sft_generation = generate_response(
            row["prompt"],
            sft_tokenizer,
            sft_model,
            sft_device,
            args.max_length,
            args.max_new_tokens,
        )
        generation_rows.append(
            {
                "category": row["category"],
                "source_id": row["source_id"],
                "prompt": row["prompt"],
                "reference": row["completion"],
                "base_generation": base_generation,
                "sft_generation": sft_generation,
                "base_rouge_l_f1": rouge_l_f1(base_generation, row["completion"]),
                "sft_rouge_l_f1": rouge_l_f1(sft_generation, row["completion"]),
                "base_copy_rate": copy_rate(base_generation, row["prompt"]),
                "sft_copy_rate": copy_rate(sft_generation, row["prompt"]),
            }
        )

    evaluation = {
        "base_model": args.base_model,
        "sft_model": args.sft_model,
        "dataset_dir": args.dataset_dir,
        "base": base_loss,
        "sft": sft_loss,
        "base_generation": summarize_generation_metrics(generation_rows, "base"),
        "sft_generation": summarize_generation_metrics(generation_rows, "sft"),
    }

    write_json(output_dir / "evaluation.json", evaluation)
    write_jsonl(output_dir / "generations.jsonl", generation_rows)
    write_report(output_dir / "analysis_report.md", evaluation, generation_rows)

    print(f"Saved evaluation metrics to {output_dir / 'evaluation.json'}")
    print(f"Saved generations to {output_dir / 'generations.jsonl'}")
    print(f"Saved report to {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
