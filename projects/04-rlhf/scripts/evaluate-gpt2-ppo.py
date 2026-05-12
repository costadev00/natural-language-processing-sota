#!/usr/bin/env python3
"""Evaluate the PPO GPT-2 policy against the SFT GPT-2 policy."""

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
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 PPO outputs against SFT.")
    parser.add_argument("--ppo-dataset-dir", default="rlhf_dolly_datasets/ppo")
    parser.add_argument("--sft-dataset-dir", default="rlhf_dolly_datasets/sft")
    parser.add_argument("--sft-model", default="sft_gpt2/model")
    parser.add_argument("--ppo-model", default="ppo_gpt2/model")
    parser.add_argument("--reward-model", default="reward_gpt2/model")
    parser.add_argument("--output-dir", default="ppo_gpt2/evaluation")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--samples-per-category", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--reward-batch-size", type=int, default=16)
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


def prepare_tokenizer(model_name_or_path: str, padding_side: str = "right") -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    tokenizer.model_max_length = 10**9
    return tokenizer


def tokenize_examples(examples: dict[str, list[str]], tokenizer: Any, max_length: int) -> dict[str, list[Any]]:
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    eos = tokenizer.eos_token or ""
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        prompt_prefix = f"{prompt}\n\n"
        prompt_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(f"{completion}{eos}", add_special_tokens=False)["input_ids"]
        if len(completion_ids) >= max_length:
            input_ids = completion_ids[:max_length]
            prompt_length = 0
        else:
            prompt_budget = max_length - len(completion_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
            input_ids = prompt_ids + completion_ids
            prompt_length = len(prompt_ids)
        labels = list(input_ids)
        labels[:prompt_length] = [-100] * prompt_length
        input_ids_batch.append(input_ids)
        attention_mask_batch.append([1] * len(input_ids))
        labels_batch.append(labels)
    return {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "labels": labels_batch}


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
    return {"loss": loss, "perplexity": float(math.exp(loss)), "samples": len(eval_dataset)}


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
            curr.append(prev[idx - 1] + 1 if pred_token == ref_token else max(curr[-1], prev[idx]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def copy_rate(prediction: str, prompt: str) -> float:
    pred_tokens = words(prediction)
    if not pred_tokens:
        return 0.0
    prompt_tokens = set(words(prompt))
    return sum(token in prompt_tokens for token in pred_tokens) / len(pred_tokens)


def encode_prompt_response(prompt: str, response: str, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(f"{prompt}\n\n", add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(f"{response}{eos}", add_special_tokens=False)["input_ids"]
    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
    else:
        input_ids = prompt_ids[-(max_length - len(response_ids)) :] + response_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def encode_prompt_response_with_start(
    prompt: str,
    response: str,
    tokenizer: Any,
    max_length: int,
) -> tuple[dict[str, list[int]], int]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(f"{prompt}\n\n", add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(f"{response}{eos}", add_special_tokens=False)["input_ids"]
    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
        response_start = 0
    else:
        prompt_window = prompt_ids[-(max_length - len(response_ids)) :]
        input_ids = prompt_window + response_ids
        response_start = len(prompt_window)
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}, response_start


def score_rewards(
    prompts: list[str],
    responses: list[str],
    reward_tokenizer: Any,
    reward_model: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> list[float]:
    scores = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_responses = responses[start : start + batch_size]
        encoded = [
            encode_prompt_response(prompt, response, reward_tokenizer, max_length)
            for prompt, response in zip(batch_prompts, batch_responses)
        ]
        batch = reward_tokenizer.pad(encoded, padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            logits = reward_model(**batch).logits.squeeze(-1)
        scores.extend(float(value) for value in logits.detach().cpu().tolist())
    return scores


def load_generation_model(model_name_or_path: str) -> tuple[Any, Any, torch.device]:
    tokenizer = prepare_tokenizer(model_name_or_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_response(prompt: str, tokenizer: Any, model: Any, device: torch.device, max_length: int, max_new_tokens: int) -> str:
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


def selected_logprobs(model: Any, tokenizer: Any, prompt: str, response: str, device: torch.device, max_length: int) -> list[float]:
    encoded, response_start = encode_prompt_response_with_start(prompt, response, tokenizer, max_length)
    input_ids = torch.tensor([encoded["input_ids"]], device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(0).squeeze(-1)
    response_logprob_start = max(response_start - 1, 0)
    return [float(value) for value in logprobs[response_logprob_start:].detach().cpu().tolist()]


def select_generation_examples(dataset_dict: DatasetDict, samples_per_category: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_dict["test"]:
        grouped[row["category"]].append(row)
    rng = random.Random(seed)
    examples = []
    for category in sorted(grouped):
        rows = grouped[category]
        rng.shuffle(rows)
        examples.extend(rows[:samples_per_category])
    return examples


def summarize_generation_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    rouge_values = [row[f"{prefix}_rouge_l_f1"] for row in rows]
    copy_values = [row[f"{prefix}_copy_rate"] for row in rows]
    lengths = [len(words(row[f"{prefix}_generation"])) for row in rows]
    rewards = [row[f"{prefix}_reward"] for row in rows]
    return {
        "rouge_l_f1_mean": float(sum(rouge_values) / len(rouge_values)) if rouge_values else 0.0,
        "copy_rate_mean": float(sum(copy_values) / len(copy_values)) if copy_values else 0.0,
        "response_words_mean": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "reward_mean": float(sum(rewards) / len(rewards)) if rewards else 0.0,
    }


def truncate(text: str, limit: int = 600) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value: Any, precision: int = 4) -> str:
    return f"`{float(value):.{precision}f}`" if isinstance(value, int | float) else "`n/a`"


def write_report(path: Path, evaluation: dict[str, Any], generations: list[dict[str, Any]]) -> None:
    reward_delta = evaluation["ppo_generation"]["reward_mean"] - evaluation["sft_generation"]["reward_mean"]
    loss_delta = evaluation["ppo"]["loss"] - evaluation["sft"]["loss"]
    train_metrics = evaluation.get("training_metrics", {})
    run_config = evaluation.get("training_config", {})
    lines = [
        "# GPT-2 PPO Evaluation",
        "",
        "## Summary",
        "",
        "- PPO policy was initialized from `sft_gpt2/model` and optimized against `reward_gpt2/model`.",
        "- The reward model was trained on synthetic proxy preferences, so this measures RM optimization rather than human alignment.",
        f"- SFT test loss: `{evaluation['sft']['loss']:.4f}`; perplexity: `{evaluation['sft']['perplexity']:.2f}`.",
        f"- PPO test loss: `{evaluation['ppo']['loss']:.4f}`; perplexity: `{evaluation['ppo']['perplexity']:.2f}`.",
        f"- PPO minus SFT loss delta: `{loss_delta:.4f}`.",
        f"- PPO reward delta on qualitative generations: `{reward_delta:.4f}`.",
        f"- PPO-vs-SFT reward win rate: `{evaluation['ppo_vs_sft_reward_win_rate']:.4f}`.",
        f"- Approx KL on PPO generations vs SFT: `{evaluation['ppo_generation']['approx_kl_vs_sft']:.4f}`.",
        f"- Generation sample size: `{evaluation['generation_sample_size']}` examples.",
        "",
        "## Dataset",
        "",
        f"- PPO prompt source: `{evaluation['ppo_dataset_dir']}`.",
        f"- Dolly/SFT reference source: `{evaluation['sft_dataset_dir']}`.",
        "- PPO prompts are prompt-only; references are joined from the SFT dataset by `source_id` for evaluation.",
        "",
        "| Split | PPO rows | SFT rows |",
        "| --- | ---: | ---: |",
    ]
    for split in ["train", "validation", "test"]:
        lines.append(
            f"| {split} | {evaluation['ppo_split_rows'].get(split, 0)} | {evaluation['sft_split_rows'].get(split, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Training Configuration",
            "",
            "| Setting | Value |",
            "| --- | --- |",
            f"| Policy initialization | `{run_config.get('policy_model', evaluation['sft_model'])}` |",
            f"| Frozen reference model | `{run_config.get('policy_model', evaluation['sft_model'])}` |",
            f"| Reward model | `{run_config.get('reward_model', evaluation['reward_model'])}` |",
            f"| Output model | `{evaluation['ppo_model']}` |",
            f"| PPO epochs | `{run_config.get('num_ppo_epochs', 'n/a')}` |",
            f"| Max new tokens | `{run_config.get('max_new_tokens', 'n/a')}` |",
            f"| Learning rate | `{run_config.get('learning_rate', 'n/a')}` |",
            f"| KL coefficient | `{run_config.get('kl_coef', 'n/a')}` |",
            f"| Clip range | `{run_config.get('clip_range', 'n/a')}` |",
            f"| Per-device batch | `{run_config.get('per_device_batch_size', 'n/a')}` |",
            f"| GPUs/processes | `{run_config.get('accelerator_processes', 'n/a')}` |",
            f"| Precision | `bf16 via accelerate launch` |",
            "",
            "## Training Diagnostics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| steps | `{train_metrics.get('steps', 'n/a')}` |",
            f"| train samples | `{train_metrics.get('train_samples', 'n/a')}` |",
            f"| reward mean | {format_metric(train_metrics.get('reward_mean_mean'))} |",
            f"| last reward mean | {format_metric(train_metrics.get('reward_mean_last'))} |",
            f"| selected-token KL mean | {format_metric(train_metrics.get('kl_mean_mean'))} |",
            f"| selected-token abs KL mean | {format_metric(train_metrics.get('kl_abs_mean_mean'))} |",
            f"| clip fraction mean | {format_metric(train_metrics.get('clip_fraction_mean'))} |",
            f"| ratio mean | {format_metric(train_metrics.get('ratio_mean'))} |",
            f"| loss mean | {format_metric(train_metrics.get('loss_mean'))} |",
            "",
            "The training diagnostics show an aggressive PPO update: high clip fraction and very large probability ratios appeared late in training. The final model still loads and evaluates, but these signals argue for a smaller LR, stronger KL control, or fewer/shorter rollouts before treating this as a stable RLHF policy.",
            "",
            "## Test Metrics",
            "",
            "| Metric | SFT | PPO | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| loss | `{evaluation['sft']['loss']:.4f}` | `{evaluation['ppo']['loss']:.4f}` | `{loss_delta:.4f}` |",
            f"| perplexity | `{evaluation['sft']['perplexity']:.2f}` | `{evaluation['ppo']['perplexity']:.2f}` | `{evaluation['ppo']['perplexity'] - evaluation['sft']['perplexity']:.2f}` |",
            "",
        ]
    )
    lines.extend(
        [
            "## Generation Metrics",
            "",
            "| Model | Reward mean | ROUGE-L F1 mean | Prompt copy rate | Mean response words |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| GPT-2 SFT | {evaluation['sft_generation']['reward_mean']:.4f} | "
                f"{evaluation['sft_generation']['rouge_l_f1_mean']:.4f} | "
                f"{evaluation['sft_generation']['copy_rate_mean']:.4f} | "
                f"{evaluation['sft_generation']['response_words_mean']:.1f} |"
            ),
            (
                f"| GPT-2 PPO | {evaluation['ppo_generation']['reward_mean']:.4f} | "
                f"{evaluation['ppo_generation']['rouge_l_f1_mean']:.4f} | "
                f"{evaluation['ppo_generation']['copy_rate_mean']:.4f} | "
                f"{evaluation['ppo_generation']['response_words_mean']:.1f} |"
            ),
            "",
            "## Qualitative Notes",
            "",
            "- PPO increased mean reward on the sampled generations, but the reward win rate is only tied with SFT at this sample size.",
            "- Test loss/perplexity worsened slightly, so PPO did not improve next-token fit to Dolly references.",
            "- ROUGE-L stayed nearly flat, while copy rate increased slightly; the qualitative examples still show repetition and shallow instruction following.",
            "- This report is downstream of a synthetic reward model; human preference evaluation would be needed before treating PPO as alignment progress.",
            "",
            "## Examples",
            "",
        ]
    )
    for row in generations[: min(8, len(generations))]:
        lines.extend(
            [
                f"### {row['category']} / {row['source_id']}",
                "",
                f"**Prompt:** {truncate(row['prompt'])}",
                "",
                f"**Reference:** {truncate(row['reference'])}",
                "",
                f"**SFT reward:** `{row['sft_reward']:.4f}`; **SFT:** {truncate(row['sft_generation'])}",
                "",
                f"**PPO reward:** `{row['ppo_reward']:.4f}`; **PPO:** {truncate(row['ppo_generation'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            (
                "The PPO run completed the full engineering loop: prompt-only dataset, SFT policy, frozen SFT reference, "
                "synthetic GPT-2 reward model, local PPO checkpoint, and comparative evaluation."
            ),
            "",
            (
                "The measured outcome is mixed. PPO improved the sampled RM reward by "
                f"`{reward_delta:.4f}`, but only tied SFT on pairwise RM wins and slightly worsened test perplexity. "
                "Given the unstable training diagnostics, this checkpoint is best treated as a working PPO prototype, "
                "not a clearly better aligned policy."
            ),
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

    sft_dataset_dict = load_from_disk(args.sft_dataset_dir)
    ppo_dataset_dict = load_from_disk(args.ppo_dataset_dir)

    sft_loss = evaluate_loss(args.sft_model, sft_dataset_dict, output_dir, args, "sft")
    ppo_loss = evaluate_loss(args.ppo_model, sft_dataset_dict, output_dir, args, "ppo")

    examples = select_generation_examples(ppo_dataset_dict, args.samples_per_category, args.seed)
    references_by_source_id = {row["source_id"]: row for row in sft_dataset_dict["test"]}
    sft_tokenizer, sft_model, sft_device = load_generation_model(args.sft_model)
    ppo_tokenizer, ppo_model, ppo_device = load_generation_model(args.ppo_model)

    reward_tokenizer = prepare_tokenizer(args.reward_model)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.to(sft_device)
    reward_model.eval()

    generation_rows = []
    for row in examples:
        reference_row = references_by_source_id.get(row["source_id"])
        reference = reference_row["completion"] if reference_row is not None else ""
        sft_generation = generate_response(row["prompt"], sft_tokenizer, sft_model, sft_device, args.max_length, args.max_new_tokens)
        ppo_generation = generate_response(row["prompt"], ppo_tokenizer, ppo_model, ppo_device, args.max_length, args.max_new_tokens)
        rewards = score_rewards(
            [row["prompt"], row["prompt"]],
            [sft_generation, ppo_generation],
            reward_tokenizer,
            reward_model,
            sft_device,
            args.max_length,
            args.reward_batch_size,
        )
        ppo_logprobs = selected_logprobs(ppo_model, ppo_tokenizer, row["prompt"], ppo_generation, ppo_device, args.max_length)
        sft_on_ppo_logprobs = selected_logprobs(sft_model, sft_tokenizer, row["prompt"], ppo_generation, sft_device, args.max_length)
        min_len = min(len(ppo_logprobs), len(sft_on_ppo_logprobs))
        approx_kl = (
            sum(ppo_logprobs[idx] - sft_on_ppo_logprobs[idx] for idx in range(min_len)) / min_len
            if min_len
            else 0.0
        )
        generation_rows.append(
            {
                "category": row["category"],
                "source_id": row["source_id"],
                "prompt": row["prompt"],
                "reference": reference,
                "sft_generation": sft_generation,
                "ppo_generation": ppo_generation,
                "sft_reward": rewards[0],
                "ppo_reward": rewards[1],
                "ppo_approx_kl_vs_sft": approx_kl,
                "sft_rouge_l_f1": rouge_l_f1(sft_generation, reference),
                "ppo_rouge_l_f1": rouge_l_f1(ppo_generation, reference),
                "sft_copy_rate": copy_rate(sft_generation, row["prompt"]),
                "ppo_copy_rate": copy_rate(ppo_generation, row["prompt"]),
            }
        )

    ppo_root = output_dir.parent
    metrics_dir = ppo_root / "metrics"
    ppo_generation_summary = summarize_generation_metrics(generation_rows, "ppo")
    ppo_generation_summary["approx_kl_vs_sft"] = float(
        sum(row["ppo_approx_kl_vs_sft"] for row in generation_rows) / len(generation_rows)
    ) if generation_rows else 0.0
    evaluation = {
        "ppo_dataset_dir": args.ppo_dataset_dir,
        "sft_dataset_dir": args.sft_dataset_dir,
        "sft_model": args.sft_model,
        "ppo_model": args.ppo_model,
        "reward_model": args.reward_model,
        "warning": "Reward model is synthetic/proxy, not human preference labels.",
        "ppo_split_rows": {split: len(ppo_dataset_dict[split]) for split in ppo_dataset_dict},
        "sft_split_rows": {split: len(sft_dataset_dict[split]) for split in sft_dataset_dict},
        "training_config": read_json_if_exists(metrics_dir / "run_config.json"),
        "training_metrics": read_json_if_exists(metrics_dir / "train_metrics.json"),
        "sft": sft_loss,
        "ppo": ppo_loss,
        "sft_generation": summarize_generation_metrics(generation_rows, "sft"),
        "ppo_generation": ppo_generation_summary,
        "ppo_vs_sft_reward_win_rate": float(
            sum(row["ppo_reward"] > row["sft_reward"] for row in generation_rows) / len(generation_rows)
        ) if generation_rows else 0.0,
        "generation_sample_size": len(generation_rows),
    }

    write_json(output_dir / "evaluation.json", evaluation)
    write_jsonl(output_dir / "generations.jsonl", generation_rows)
    write_report(output_dir / "analysis_report.md", evaluation, generation_rows)

    print(f"Saved evaluation metrics to {output_dir / 'evaluation.json'}")
    print(f"Saved generations to {output_dir / 'generations.jsonl'}")
    print(f"Saved report to {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
