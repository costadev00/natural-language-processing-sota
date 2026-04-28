#!/usr/bin/env python3
"""Conservative PPO fine-tuning for the local GPT-2 SFT policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO on GPT-2 SFT with the local reward model.")
    parser.add_argument("--dataset-dir", default="rlhf_dolly_datasets/ppo")
    parser.add_argument("--policy-model", default="sft_gpt2/model")
    parser.add_argument("--reward-model", default="reward_gpt2/model")
    parser.add_argument("--output-dir", default="ppo_gpt2")
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-ppo-epochs", type=int, default=1)
    parser.add_argument("--ppo-update-epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    return parser.parse_args()


class PolicyWithValue(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        values = self.value_head(outputs.hidden_states[-1]).squeeze(-1)
        return outputs.logits, values


@dataclass
class RolloutBatch:
    full_input_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    prompt_width: int
    response_width: int
    response_texts: list[str]


def guard_against_single_process_dataparallel() -> None:
    is_distributed = "LOCAL_RANK" in os.environ or "RANK" in os.environ
    if torch.cuda.device_count() > 1 and not is_distributed:
        raise RuntimeError(
            "Multiple GPUs are visible, but this script was launched without DDP. "
            "Use: .venv/bin/accelerate launch --multi_gpu --num_processes 4 train_gpt2_ppo.py ..."
        )


def collate_prompts(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "prompt": [row["prompt"] for row in rows],
        "category": [row["category"] for row in rows],
        "source_id": [row["source_id"] for row in rows],
        "has_context": [row["has_context"] for row in rows],
    }


def subset_train(dataset_dict: DatasetDict, max_samples: int | None, seed: int):
    dataset = dataset_dict["train"]
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def prepare_tokenizer(model_name_or_path: str, padding_side: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    tokenizer.model_max_length = 10**9
    return tokenizer


def encode_prompt_response(prompt: str, response: str, tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(f"{prompt}\n\n", add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(f"{response}{eos}", add_special_tokens=False)["input_ids"]
    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
    else:
        input_ids = prompt_ids[-(max_length - len(response_ids)) :] + response_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def gather_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def response_logprobs_and_values(
    model: PolicyWithValue | nn.Module,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    prompt_width: int,
    response_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, values = model(full_input_ids, full_attention_mask)
    logprobs = gather_logprobs(logits, full_input_ids)
    start = max(prompt_width - 1, 0)
    end = start + response_width
    return logprobs[:, start:end], values[:, :-1][:, start:end]


def response_logprobs_causal(
    model: Any,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    prompt_width: int,
    response_width: int,
) -> torch.Tensor:
    outputs = model(input_ids=full_input_ids, attention_mask=full_attention_mask, use_cache=False)
    logprobs = gather_logprobs(outputs.logits, full_input_ids)
    start = max(prompt_width - 1, 0)
    return logprobs[:, start : start + response_width]


def score_reward_model(
    prompts: list[str],
    responses: list[str],
    reward_tokenizer: Any,
    reward_model: Any,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    encoded_rows = [
        encode_prompt_response(prompt, response, reward_tokenizer, max_length)
        for prompt, response in zip(prompts, responses)
    ]
    batch = reward_tokenizer.pad(encoded_rows, padding=True, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.no_grad():
        rewards = reward_model(**batch).logits.squeeze(-1)
    return rewards.float()


def decode_responses(tokenizer: Any, generated_ids: torch.Tensor, response_mask: torch.Tensor) -> list[str]:
    responses = []
    for ids, mask in zip(generated_ids, response_mask):
        valid_ids = ids[mask.bool()]
        responses.append(tokenizer.decode(valid_ids, skip_special_tokens=True).strip())
    return responses


def make_rollout(
    prompts: list[str],
    policy: Any,
    ref_model: Any,
    reward_model: Any,
    tokenizer: Any,
    reward_tokenizer: Any,
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> RolloutBatch:
    generation_prompts = [f"{prompt}\n\n" for prompt in prompts]
    encoded = tokenizer(
        generation_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        add_special_tokens=False,
    )
    encoded = {key: value.to(accelerator.device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    unwrapped_policy = accelerator.unwrap_model(policy).base_model
    unwrapped_policy.eval()
    with torch.no_grad():
        output_ids = unwrapped_policy.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[:, prompt_width:]
    response_width = generated_ids.shape[1]
    response_mask = generated_ids.ne(tokenizer.pad_token_id).long()
    empty_rows = response_mask.sum(dim=1).eq(0)
    if empty_rows.any():
        generated_ids[empty_rows, 0] = tokenizer.eos_token_id
        response_mask[empty_rows, 0] = 1

    full_input_ids = output_ids
    full_attention_mask = torch.cat([encoded["attention_mask"], response_mask], dim=1)
    response_texts = decode_responses(tokenizer, generated_ids, response_mask)
    fallback_text = "I do not have enough information to answer this question."
    response_texts = [text if text else fallback_text for text in response_texts]

    with torch.no_grad():
        old_logprobs, old_values = response_logprobs_and_values(
            policy,
            full_input_ids,
            full_attention_mask,
            prompt_width,
            response_width,
        )
        ref_logprobs = response_logprobs_causal(
            ref_model,
            full_input_ids,
            full_attention_mask,
            prompt_width,
            response_width,
        )
        rewards = score_reward_model(
            prompts,
            response_texts,
            reward_tokenizer,
            reward_model,
            accelerator.device,
            args.max_prompt_length + args.max_new_tokens,
        )
    unwrapped_policy.train()

    return RolloutBatch(
        full_input_ids=full_input_ids,
        full_attention_mask=full_attention_mask,
        response_mask=response_mask.float(),
        old_logprobs=old_logprobs.detach(),
        ref_logprobs=ref_logprobs.detach(),
        old_values=old_values.detach(),
        rewards=rewards.detach(),
        prompt_width=prompt_width,
        response_width=response_width,
        response_texts=response_texts,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_returns(
    rewards: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_coef: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_kl = (old_logprobs - ref_logprobs) * response_mask
    token_rewards = -kl_coef * token_kl
    lengths = response_mask.sum(dim=1).long().clamp_min(1)
    for row_idx, length in enumerate(lengths.tolist()):
        token_rewards[row_idx, length - 1] += rewards[row_idx]

    returns = torch.zeros_like(token_rewards)
    running = torch.zeros(token_rewards.shape[0], device=token_rewards.device)
    for idx in reversed(range(token_rewards.shape[1])):
        running = token_rewards[:, idx] + running
        returns[:, idx] = running
    returns = returns * response_mask
    return returns, token_kl


def normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = advantages[mask.bool()]
    if valid.numel() < 2:
        return advantages * mask
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(1e-6)
    return ((advantages - mean) / std) * mask


def ppo_update(
    policy: Any,
    rollout: RolloutBatch,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> dict[str, float]:
    returns, token_kl = compute_returns(
        rollout.rewards,
        rollout.old_logprobs,
        rollout.ref_logprobs,
        rollout.response_mask,
        args.kl_coef,
    )
    advantages = normalize_advantages(returns - rollout.old_values, rollout.response_mask)

    metrics: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "loss": [],
        "ratio": [],
        "clip_fraction": [],
    }

    for _ in range(args.ppo_update_epochs):
        new_logprobs, values = response_logprobs_and_values(
            policy,
            rollout.full_input_ids,
            rollout.full_attention_mask,
            rollout.prompt_width,
            rollout.response_width,
        )
        ratio = torch.exp(new_logprobs - rollout.old_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
        policy_loss = -masked_mean(torch.minimum(unclipped, clipped), rollout.response_mask)
        value_loss = masked_mean((values - returns) ** 2, rollout.response_mask)
        loss = policy_loss + args.value_coef * value_loss

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()

        clip_fraction = ((ratio - 1.0).abs() > args.clip_range).float()
        metrics["policy_loss"].append(float(policy_loss.detach().float().item()))
        metrics["value_loss"].append(float(value_loss.detach().float().item()))
        metrics["loss"].append(float(loss.detach().float().item()))
        metrics["ratio"].append(float(masked_mean(ratio.detach(), rollout.response_mask).float().item()))
        metrics["clip_fraction"].append(float(masked_mean(clip_fraction.detach(), rollout.response_mask).float().item()))

    summary = {key: sum(values) / len(values) for key, values in metrics.items()}
    summary["reward_mean"] = float(rollout.rewards.mean().detach().float().item())
    summary["response_tokens_mean"] = float(rollout.response_mask.sum(dim=1).mean().detach().float().item())
    summary["kl_mean"] = float(masked_mean(token_kl.detach(), rollout.response_mask).float().item())
    summary["kl_abs_mean"] = float(masked_mean(token_kl.detach().abs(), rollout.response_mask).float().item())
    summary["non_score_reward_mean"] = float(masked_mean((-args.kl_coef * token_kl).detach(), rollout.response_mask).float().item())
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    guard_against_single_process_dataparallel()
    set_seed(args.seed)
    random.seed(args.seed)

    accelerator = Accelerator()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This PPO run is expected to train on GPU.")

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    if accelerator.is_main_process:
        metrics_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = prepare_tokenizer(args.policy_model, padding_side="left")
    reward_tokenizer = prepare_tokenizer(args.reward_model, padding_side="right")

    policy = PolicyWithValue(args.policy_model)
    policy.base_model.config.pad_token_id = tokenizer.pad_token_id
    policy.base_model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(args.policy_model)
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.use_cache = False
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    dataset_dict = load_from_disk(args.dataset_dir)
    train_dataset = subset_train(dataset_dict, args.max_train_samples, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_prompts,
        num_workers=args.dataloader_num_workers,
    )

    total_steps = math.ceil(len(train_dataset) / (args.per_device_batch_size * max(accelerator.num_processes, 1)))
    total_steps *= args.num_ppo_epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    policy, optimizer, train_loader, scheduler = accelerator.prepare(policy, optimizer, train_loader, scheduler)
    ref_model.to(accelerator.device)
    reward_model.to(accelerator.device)

    if accelerator.is_main_process:
        run_config = {
            "dataset_dir": args.dataset_dir,
            "policy_model": args.policy_model,
            "reward_model": args.reward_model,
            "output_dir": args.output_dir,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_ppo_epochs": args.num_ppo_epochs,
            "ppo_update_epochs": args.ppo_update_epochs,
            "per_device_batch_size": args.per_device_batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "clip_range": args.clip_range,
            "kl_coef": args.kl_coef,
            "value_coef": args.value_coef,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
            "cuda_device_count": torch.cuda.device_count(),
            "accelerator_processes": accelerator.num_processes,
            "total_steps": total_steps,
            "preference_dataset_warning": "Reward model was trained on synthetic proxy preferences.",
        }
        write_json(metrics_dir / "run_config.json", run_config)
        rollout_log = metrics_dir / "rollout_metrics.jsonl"
        if rollout_log.exists():
            rollout_log.unlink()

    global_step = 0
    aggregate: dict[str, list[float]] = {}
    progress = tqdm(total=total_steps, disable=not accelerator.is_main_process, desc="PPO training")

    for epoch in range(args.num_ppo_epochs):
        for batch in train_loader:
            prompts = list(batch["prompt"])
            rollout = make_rollout(prompts, policy, ref_model, reward_model, tokenizer, reward_tokenizer, accelerator, args)
            metrics = ppo_update(policy, rollout, optimizer, accelerator, args)
            scheduler.step()
            global_step += 1

            gathered_metrics = {}
            for key, value in metrics.items():
                tensor = torch.tensor(value, device=accelerator.device)
                gathered = accelerator.gather_for_metrics(tensor)
                gathered_metrics[key] = float(gathered.float().mean().item())
                aggregate.setdefault(key, []).append(gathered_metrics[key])

            if accelerator.is_main_process:
                row = {"step": global_step, "epoch": epoch + 1, **gathered_metrics}
                append_jsonl(metrics_dir / "rollout_metrics.jsonl", row)
                if global_step % args.logging_steps == 0 or global_step == 1:
                    progress.set_postfix(
                        reward=f"{gathered_metrics['reward_mean']:.3f}",
                        kl=f"{gathered_metrics['kl_mean']:.4f}",
                        loss=f"{gathered_metrics['loss']:.3f}",
                    )
            progress.update(1)

    progress.close()
    accelerator.wait_for_everyone()

    summary = {
        "steps": global_step,
        "train_samples": len(train_dataset),
        "distributed_processes": accelerator.num_processes,
    }
    for key, values in aggregate.items():
        summary[f"{key}_mean"] = float(sum(values) / len(values)) if values else 0.0
        summary[f"{key}_last"] = float(values[-1]) if values else 0.0

    unwrapped_policy = accelerator.unwrap_model(policy)
    model_dir = output_dir / "model"
    if accelerator.is_main_process:
        unwrapped_policy.base_model.save_pretrained(str(model_dir), save_function=accelerator.save)
        tokenizer.save_pretrained(str(model_dir))
        torch.save(unwrapped_policy.value_head.state_dict(), output_dir / "value_head.pt")
        write_json(metrics_dir / "train_metrics.json", summary)
        print(f"Saved PPO policy to {model_dir}")
        print(f"Saved PPO metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
