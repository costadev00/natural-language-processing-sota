#!/usr/bin/env python3
"""Build synthetic preference pairs for a GPT-2 reward model.

The original rm_schema dataset is a labeling pool, not a trainable preference
dataset. This script creates a clearly marked proxy dataset by treating the
Dolly reference response as chosen and an SFT GPT-2 generation as rejected.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic RM preference pairs from Dolly RM schema.")
    parser.add_argument("--rm-schema-dir", default="rlhf_dolly_datasets/rm_schema")
    parser.add_argument("--output-dir", default="rlhf_dolly_datasets/rm_synthetic")
    parser.add_argument("--rejected-model", default="sft_gpt2/model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def prepare_model(model_name_or_path: str, tokenizer: Any) -> tuple[Any, torch.device]:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model.to(device)
    model.eval()
    return model, device


def subset_split(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def generate_rejections(
    prompts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> list[str]:
    max_input_tokens = max(16, args.max_length - args.max_new_tokens)
    generation_prompts = [f"{prompt}\n\n" for prompt in prompts]
    encoded = tokenizer(
        generation_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_width = encoded["input_ids"].shape[1]
    generations = []
    for row_idx in range(len(prompts)):
        generated_ids = output_ids[row_idx, prompt_width:]
        generations.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return generations


def normalize_rejected(generated: str, chosen: str) -> tuple[str, bool]:
    rejected = " ".join(generated.split())
    fallback_used = False
    if not rejected:
        rejected = "I do not have enough information to answer this question."
        fallback_used = True
    if rejected == " ".join(chosen.split()):
        rejected = f"{rejected} I am not certain about the details."
        fallback_used = True
    return rejected, fallback_used


def build_split(
    split: str,
    dataset: Dataset,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Dataset, dict[str, Any]]:
    rows = []
    fallback_count = 0
    split_seed_offset = SPLITS.index(split) * 1_000_000

    for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"Generating {split} rejections"):
        batch = dataset[start : start + args.batch_size]
        prompts = list(batch["prompt"])
        torch.manual_seed(args.seed + split_seed_offset + start)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + split_seed_offset + start)
        generated_responses = generate_rejections(prompts, tokenizer, model, device, args)

        for idx, generated in enumerate(generated_responses):
            chosen = str(batch["reference_response"][idx]).strip()
            rejected, fallback_used = normalize_rejected(generated, chosen)
            fallback_count += int(fallback_used)
            rows.append(
                {
                    "prompt": str(batch["prompt"][idx]),
                    "chosen": chosen,
                    "rejected": rejected,
                    "reference_response": chosen,
                    "category": str(batch["category"][idx]),
                    "source_id": str(batch["source_id"][idx]),
                    "has_context": bool(batch["has_context"][idx]),
                    "ready_for_rm": True,
                    "preference_source": "synthetic_sft_generation",
                    "rejected_model": args.rejected_model,
                }
            )

    return Dataset.from_list(rows), {"rows": len(rows), "fallback_rejections": fallback_count}


def validate_dataset(dataset_dict: DatasetDict) -> None:
    source_ids_by_split = {}
    for split in SPLITS:
        dataset = dataset_dict[split]
        if len(dataset) == 0:
            raise ValueError(f"{split} split is empty.")
        source_ids = set(dataset["source_id"])
        if len(source_ids) != len(dataset):
            raise ValueError(f"{split} split contains duplicate source_id values.")
        source_ids_by_split[split] = source_ids
        for row in dataset:
            if not row["prompt"].strip():
                raise ValueError(f"Empty prompt in {split}.")
            if not row["chosen"].strip():
                raise ValueError(f"Empty chosen response in {split}.")
            if not row["rejected"].strip():
                raise ValueError(f"Empty rejected response in {split}.")
            if not row["ready_for_rm"]:
                raise ValueError(f"ready_for_rm must be true in {split}.")

    for idx, split in enumerate(SPLITS):
        for other in SPLITS[idx + 1 :]:
            overlap = source_ids_by_split[split] & source_ids_by_split[other]
            if overlap:
                raise ValueError(f"{split} and {other} share source_id values.")


def write_metadata(output_dir: Path, args: argparse.Namespace, stats: dict[str, Any]) -> None:
    payload = {
        "dataset_type": "synthetic_reward_model_preferences",
        "chosen_source": "databricks_dolly_reference_response",
        "rejected_source": "sampled_sft_gpt2_generation",
        "warning": "This is a proxy RM dataset, not human preference data.",
        "rm_schema_dir": args.rm_schema_dir,
        "rejected_model": args.rejected_model,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
        "stats": stats,
    }
    (output_dir / "synthetic_preference_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    schema = load_from_disk(args.rm_schema_dir)
    tokenizer = prepare_tokenizer(args.rejected_model)
    model, device = prepare_model(args.rejected_model, tokenizer)

    outputs = {}
    stats = {}
    for split in SPLITS:
        split_dataset = subset_split(schema[split], args.max_samples_per_split, args.seed)
        outputs[split], stats[split] = build_split(split, split_dataset, tokenizer, model, device, args)

    dataset_dict = DatasetDict(outputs)
    validate_dataset(dataset_dict)
    dataset_dict.save_to_disk(str(output_dir))
    write_metadata(output_dir, args, stats)

    print(f"Saved synthetic RM preferences to {output_dir}")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
