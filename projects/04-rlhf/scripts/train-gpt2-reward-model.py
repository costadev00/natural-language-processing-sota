#!/usr/bin/env python3
"""Train a GPT-2 reward model on synthetic Dolly preference pairs."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 as a scalar reward model.")
    parser.add_argument("--dataset-dir", default="rlhf_dolly_datasets/rm_synthetic")
    parser.add_argument("--output-dir", default="reward_gpt2")
    parser.add_argument("--model-name", default="sft_gpt2/model")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--disable-fp16", action="store_true")
    return parser.parse_args()


@dataclass
class RewardDataCollator:
    tokenizer: Any

    def _pad(self, features: list[dict[str, Any]], prefix: str) -> dict[str, torch.Tensor]:
        rows = [
            {
                "input_ids": feature[f"{prefix}_input_ids"],
                "attention_mask": feature[f"{prefix}_attention_mask"],
            }
            for feature in features
        ]
        return self.tokenizer.pad(rows, padding=True, return_tensors="pt")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen = self._pad(features, "chosen")
        rejected = self._pad(features, "rejected")
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }


class RewardTrainer(Trainer):
    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ):
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        chosen_rewards = chosen_outputs.logits.squeeze(-1)
        rejected_rewards = rejected_outputs.logits.squeeze(-1)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards.detach(),
                "rejected_rewards": rejected_rewards.detach(),
            }
        return loss

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = loss.detach()
        if prediction_loss_only:
            return loss, None, None

        logits = torch.stack([outputs["chosen_rewards"], outputs["rejected_rewards"]], dim=-1)
        labels = torch.ones(logits.shape[0], dtype=torch.long, device=logits.device)
        return loss, logits.detach(), labels.detach()


def subset_split(dataset_dict: DatasetDict, split: str, max_samples: int | None, seed: int) -> Dataset:
    dataset = dataset_dict[split]
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


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


def tokenize_examples(examples: dict[str, list[str]], tokenizer: Any, max_length: int) -> dict[str, list[Any]]:
    chosen_input_ids = []
    chosen_attention_mask = []
    rejected_input_ids = []
    rejected_attention_mask = []

    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        chosen_encoded = encode_prompt_response(prompt, chosen, tokenizer, max_length)
        rejected_encoded = encode_prompt_response(prompt, rejected, tokenizer, max_length)
        chosen_input_ids.append(chosen_encoded["input_ids"])
        chosen_attention_mask.append(chosen_encoded["attention_mask"])
        rejected_input_ids.append(rejected_encoded["input_ids"])
        rejected_attention_mask.append(rejected_encoded["attention_mask"])

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
    }


def has_pair_tokens(example: dict[str, Any]) -> bool:
    return bool(example["chosen_input_ids"]) and bool(example["rejected_input_ids"])


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


def make_training_arguments(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    cuda_available = torch.cuda.is_available()
    use_bf16 = bool(cuda_available and torch.cuda.is_bf16_supported() and not args.disable_fp16)
    use_fp16 = bool(cuda_available and not use_bf16 and not args.disable_fp16)

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "bf16": use_bf16,
        "fp16": use_fp16,
    }
    if supports_training_arg("eval_strategy"):
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if supports_training_arg("save_strategy"):
        kwargs["save_strategy"] = "epoch"
    if supports_training_arg("logging_strategy"):
        kwargs["logging_strategy"] = "steps"
    if supports_training_arg("tf32"):
        kwargs["tf32"] = bool(cuda_available)
    if supports_training_arg("logging_dir"):
        kwargs["logging_dir"] = str(output_dir / "logs")
    return TrainingArguments(**supported_training_kwargs(kwargs))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def guard_against_trainer_dataparallel() -> None:
    is_distributed = "LOCAL_RANK" in os.environ or "RANK" in os.environ
    if torch.cuda.device_count() > 1 and not is_distributed:
        raise RuntimeError(
            "Multiple GPUs are visible, but this script was launched without DDP. "
            "Use: .venv/bin/accelerate launch --multi_gpu --num_processes 4 scripts/train-gpt2-reward-model.py ..."
        )


def prepare_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 10**9
    return tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    guard_against_trainer_dataparallel()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This reward model run is expected to train on GPU.")

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = prepare_tokenizer(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    dataset_dict = load_from_disk(args.dataset_dir)
    train_dataset = subset_split(dataset_dict, "train", args.max_train_samples, args.seed)
    eval_dataset = subset_split(dataset_dict, "validation", args.max_eval_samples, args.seed)

    tokenized_train = train_dataset.map(
        lambda batch: tokenize_examples(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train pairs",
    ).filter(has_pair_tokens, desc="Filtering train pairs")
    tokenized_eval = eval_dataset.map(
        lambda batch: tokenize_examples(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation pairs",
    ).filter(has_pair_tokens, desc="Filtering validation pairs")

    training_args = make_training_arguments(args, output_dir)
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=RewardDataCollator(tokenizer),
        **trainer_tokenizer_kwarg(tokenizer),
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(tokenized_train)
    if trainer.is_world_process_zero():
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        save_json(metrics_dir / "train_metrics.json", train_metrics)

    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        eval_metrics["eval_pair_loss"] = eval_metrics["eval_loss"]
        eval_metrics["eval_pair_perplexity_like"] = math.exp(eval_metrics["eval_loss"])
    eval_metrics["eval_samples"] = len(tokenized_eval)
    if trainer.is_world_process_zero():
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        save_json(metrics_dir / "eval_metrics.json", eval_metrics)

    model_dir = output_dir / "model"
    if trainer.is_world_process_zero():
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))

    run_config = {
        "model_name": args.model_name,
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "cuda_device_count": torch.cuda.device_count(),
        "distributed_rank": int(os.environ.get("RANK", "-1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
        "bf16": bool(getattr(training_args, "bf16", False)),
        "fp16": bool(getattr(training_args, "fp16", False)),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "preference_dataset_warning": "Synthetic proxy preferences, not human labels.",
    }
    if trainer.is_world_process_zero():
        save_json(metrics_dir / "run_config.json", run_config)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        print(f"Saved reward model to {model_dir}")
        print(f"Saved metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
