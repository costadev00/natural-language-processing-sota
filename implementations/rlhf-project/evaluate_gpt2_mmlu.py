#!/usr/bin/env python3
"""Evaluate GPT-2 checkpoints on the MMLU benchmark.

The benchmark protocol follows Hendrycks et al. (2021): multiple-choice
questions, a fixed five-example dev prompt for few-shot runs, and prediction by
the model probability of answer letters A-D.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


LETTERS = ["A", "B", "C", "D"]
EXCLUDED_CONFIGS = {"all", "auxiliary_train"}

SUPER_CATEGORIES = {
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "Other",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "Social Sciences",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "Humanities",
    "global_facts": "Other",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities",
    "high_school_geography": "Social Sciences",
    "high_school_government_and_politics": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social Sciences",
    "high_school_physics": "STEM",
    "high_school_psychology": "Social Sciences",
    "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "human_aging": "Other",
    "human_sexuality": "Social Sciences",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "machine_learning": "STEM",
    "management": "Other",
    "marketing": "Other",
    "medical_genetics": "Other",
    "miscellaneous": "Other",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "nutrition": "Other",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_accounting": "Other",
    "professional_law": "Humanities",
    "professional_medicine": "Other",
    "professional_psychology": "Social Sciences",
    "public_relations": "Social Sciences",
    "security_studies": "Social Sciences",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "virology": "Other",
    "world_religions": "Humanities",
}


@dataclass
class PreparedExample:
    subject: str
    supercategory: str
    index: int
    question: str
    choices: list[str]
    answer_index: int
    input_ids: list[int]
    n_shot_used: int
    original_context_tokens: int
    context_tokens: int
    was_truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 base/SFT on MMLU.")
    parser.add_argument("--dataset-name", default="cais/mmlu")
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--sft-model", default="sft_gpt2/model")
    parser.add_argument("--output-dir", default="sft_gpt2/mmlu_evaluation")
    parser.add_argument("--models", default="base,sft", help="Comma-separated subset of: base,sft.")
    parser.add_argument("--modes", default="zero_shot,five_shot", help="Comma-separated subset of: zero_shot,five_shot.")
    parser.add_argument("--subjects", default=None, help="Comma-separated MMLU subject config names.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap per subject.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def parse_csv_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_choices(values: list[str], allowed: set[str], label: str) -> None:
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Invalid {label}: {', '.join(invalid)}. Allowed: {', '.join(sorted(allowed))}")


def subject_label(subject: str) -> str:
    return subject.replace("_", " ")


def answer_letter(answer: int) -> str:
    return LETTERS[int(answer)]


def format_mmlu_example(row: dict[str, Any], include_answer: bool) -> str:
    lines = [str(row["question"]).strip()]
    for letter, choice in zip(LETTERS, row["choices"]):
        lines.append(f"({letter}) {str(choice).strip()}")
    if include_answer:
        lines.append(f"Answer: {answer_letter(row['answer'])}")
    else:
        lines.append("Answer: ")
    return "\n".join(lines)


def make_prompt(subject: str, dev_rows: list[dict[str, Any]], test_row: dict[str, Any]) -> str:
    intro = f"The following are multiple choice questions (with answers) about {subject_label(subject)}."
    parts = [intro]
    parts.extend(format_mmlu_example(row, include_answer=True) for row in dev_rows)
    parts.append(format_mmlu_example(test_row, include_answer=False))
    return "\n\n".join(parts)


def scoring_context(prompt: str) -> str:
    # Keep the rendered prompt as "Answer: ", but score " A"/" B"/" C"/" D"
    # after the tokenizer-stable context "Answer:".
    if prompt.endswith("Answer: "):
        return prompt[:-1]
    return prompt


def prepare_prompt_input_ids(
    subject: str,
    dev_rows: list[dict[str, Any]],
    test_row: dict[str, Any],
    tokenizer: Any,
    max_length: int,
) -> tuple[list[int], int, int, int, bool]:
    working_dev_rows = list(dev_rows)
    while True:
        prompt = make_prompt(subject, working_dev_rows, test_row)
        ids = tokenizer(scoring_context(prompt), add_special_tokens=False)["input_ids"]
        if len(ids) <= max_length:
            return ids, len(working_dev_rows), len(ids), len(ids), False
        if working_dev_rows:
            working_dev_rows = working_dev_rows[:-1]
            continue
        return ids[-max_length:], 0, len(ids), max_length, True


def prepare_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name_or_path: str, device: torch.device) -> Any:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.to(device)
    model.eval()
    return model


def resolve_subjects(args: argparse.Namespace) -> list[str]:
    configs = sorted(get_dataset_config_names(args.dataset_name))
    available = [config for config in configs if config not in EXCLUDED_CONFIGS]
    requested = parse_csv_arg(args.subjects)
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Unknown MMLU subjects: {', '.join(missing)}")
        subjects = requested
    else:
        subjects = available

    missing_categories = sorted(set(subjects) - set(SUPER_CATEGORIES))
    if missing_categories:
        raise ValueError(f"Missing supercategory mapping for: {', '.join(missing_categories)}")
    return subjects


def maybe_subset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def load_subject_rows(args: argparse.Namespace, subjects: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    subject_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for subject in subjects:
        dev = load_dataset(args.dataset_name, subject, split="dev")
        test = load_dataset(args.dataset_name, subject, split="test")
        test = maybe_subset(test, args.max_test_samples, args.seed)
        subject_rows[subject] = {
            "dev": [dict(row) for row in dev],
            "test": [dict(row) for row in test],
        }
    return subject_rows


def prepare_examples(
    mode: str,
    subject_rows: dict[str, dict[str, list[dict[str, Any]]]],
    tokenizer: Any,
    max_length: int,
) -> list[PreparedExample]:
    prepared: list[PreparedExample] = []
    for subject in sorted(subject_rows):
        dev_rows = subject_rows[subject]["dev"][:5] if mode == "five_shot" else []
        for index, row in enumerate(subject_rows[subject]["test"]):
            input_ids, n_shot_used, original_tokens, context_tokens, truncated = prepare_prompt_input_ids(
                subject,
                dev_rows,
                row,
                tokenizer,
                max_length,
            )
            prepared.append(
                PreparedExample(
                    subject=subject,
                    supercategory=SUPER_CATEGORIES[subject],
                    index=index,
                    question=str(row["question"]),
                    choices=[str(choice) for choice in row["choices"]],
                    answer_index=int(row["answer"]),
                    input_ids=input_ids,
                    n_shot_used=n_shot_used,
                    original_context_tokens=original_tokens,
                    context_tokens=context_tokens,
                    was_truncated=truncated,
                )
            )
    return prepared


def option_token_ids(tokenizer: Any) -> list[int]:
    token_ids: list[int] = []
    for letter in LETTERS:
        ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Expected single token for answer {letter!r}, got token ids {ids}")
        token_ids.append(ids[0])
    return token_ids


def pad_batch(examples: list[PreparedExample], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(example.input_ids) for example in examples)
    input_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
    for row_idx, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row_idx, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row_idx, :length] = 1
    return input_ids.to(device), attention_mask.to(device)


def next_token_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)

    if hasattr(model, "transformer") and hasattr(model, "lm_head"):
        outputs = model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last_hidden = outputs.last_hidden_state[batch_indices, last_indices]
        return model.lm_head(last_hidden)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return outputs.logits[batch_indices, last_indices]


def evaluate_prepared_examples(
    model_key: str,
    model_path: str,
    mode: str,
    examples: list[PreparedExample],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    answer_token_ids = option_token_ids(tokenizer)
    rows: list[dict[str, Any]] = []
    for batch_idx, start in enumerate(range(0, len(examples), batch_size), start=1):
        batch = examples[start : start + batch_size]
        input_ids, attention_mask = pad_batch(batch, tokenizer.pad_token_id, device)
        with torch.inference_mode():
            next_logits = next_token_logits(model, input_ids, attention_mask)
            answer_logits = next_logits[:, answer_token_ids]
            answer_probs = torch.softmax(answer_logits.float(), dim=-1).cpu()

        for local_idx, example in enumerate(batch):
            probs = answer_probs[local_idx].tolist()
            prediction_index = int(max(range(len(probs)), key=lambda idx: probs[idx]))
            correct = prediction_index == example.answer_index
            rows.append(
                {
                    "model": model_key,
                    "model_path": model_path,
                    "mode": mode,
                    "subject": example.subject,
                    "subject_label": subject_label(example.subject),
                    "supercategory": example.supercategory,
                    "index": example.index,
                    "question": example.question,
                    "choices": example.choices,
                    "answer": answer_letter(example.answer_index),
                    "prediction": answer_letter(prediction_index),
                    "correct": correct,
                    "confidence": float(probs[prediction_index]),
                    "probabilities": {letter: float(probs[idx]) for idx, letter in enumerate(LETTERS)},
                    "n_shot_used": example.n_shot_used,
                    "original_context_tokens": example.original_context_tokens,
                    "context_tokens": example.context_tokens,
                    "was_truncated": example.was_truncated,
                }
            )
        if batch_idx % 100 == 0 or start + batch_size >= len(examples):
            print(f"  {model_key}/{mode}: scored {min(start + batch_size, len(examples))}/{len(examples)}", flush=True)
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def calibration_metrics(rows: list[dict[str, Any]], bins: int) -> dict[str, float]:
    if not rows:
        return {
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "calibration_gap": 0.0,
            "expected_calibration_error": 0.0,
            "rms_calibration_error": 0.0,
        }

    accuracy = mean([1.0 if row["correct"] else 0.0 for row in rows])
    avg_confidence = mean([float(row["confidence"]) for row in rows])
    ece = 0.0
    rms = 0.0
    total = len(rows)
    for bin_idx in range(bins):
        lower = bin_idx / bins
        upper = (bin_idx + 1) / bins
        if bin_idx == bins - 1:
            bin_rows = [row for row in rows if lower <= float(row["confidence"]) <= upper]
        else:
            bin_rows = [row for row in rows if lower <= float(row["confidence"]) < upper]
        if not bin_rows:
            continue
        bin_acc = mean([1.0 if row["correct"] else 0.0 for row in bin_rows])
        bin_conf = mean([float(row["confidence"]) for row in bin_rows])
        weight = len(bin_rows) / total
        ece += weight * abs(bin_conf - bin_acc)
        rms += weight * (bin_conf - bin_acc) ** 2

    return {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "calibration_gap": avg_confidence - accuracy,
        "expected_calibration_error": float(ece),
        "rms_calibration_error": float(math.sqrt(rms)),
    }


def summarize_rows(rows: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    summary = calibration_metrics(rows, bins)
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": len(rows),
        "correct": correct,
        **summary,
        "n_shot_mean": mean([float(row["n_shot_used"]) for row in rows]),
        "n_shot_min": min((int(row["n_shot_used"]) for row in rows), default=0),
        "n_shot_max": max((int(row["n_shot_used"]) for row in rows), default=0),
        "truncated_count": sum(1 for row in rows if row["was_truncated"]),
    }


def grouped_summaries(rows: list[dict[str, Any]], key: str, bins: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {group: summarize_rows(group_rows, bins) for group, group_rows in sorted(grouped.items())}


def build_evaluation(
    all_rows: dict[tuple[str, str], list[dict[str, Any]]],
    args: argparse.Namespace,
    subjects: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for (model_key, mode), rows in sorted(all_rows.items()):
        results.setdefault(model_key, {})[mode] = {
            "overall": summarize_rows(rows, args.calibration_bins),
            "by_supercategory": grouped_summaries(rows, "supercategory", args.calibration_bins),
            "by_subject": grouped_summaries(rows, "subject", args.calibration_bins),
        }

    return {
        "config": {
            "dataset_name": args.dataset_name,
            "subjects": subjects,
            "models": parse_csv_arg(args.models),
            "modes": parse_csv_arg(args.modes),
            "base_model": args.base_model,
            "sft_model": args.sft_model,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_test_samples_per_subject": args.max_test_samples,
            "seed": args.seed,
            "calibration_bins": args.calibration_bins,
        },
        "dataset": {
            "subject_count": len(subjects),
            "test_examples": None,
            "paper_reported_test_examples": 14079 if len(subjects) == 57 and args.max_test_samples is None else None,
        },
        "results": results,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def flatten_subject_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model_results in evaluation["results"].items():
        for mode, mode_results in model_results.items():
            for subject, summary in mode_results["by_subject"].items():
                rows.append(
                    {
                        "model": model_key,
                        "mode": mode,
                        "subject": subject,
                        "subject_label": subject_label(subject),
                        "supercategory": SUPER_CATEGORIES[subject],
                        **summary,
                    }
                )
    return rows


def flatten_supercategory_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model_results in evaluation["results"].items():
        for mode, mode_results in model_results.items():
            for supercategory, summary in mode_results["by_supercategory"].items():
                rows.append({"model": model_key, "mode": mode, "supercategory": supercategory, **summary})
    return rows


def flatten_calibration_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model_results in evaluation["results"].items():
        for mode, mode_results in model_results.items():
            overall = mode_results["overall"]
            rows.append({"model": model_key, "mode": mode, "group_type": "overall", "group": "all", **overall})
            for supercategory, summary in mode_results["by_supercategory"].items():
                rows.append(
                    {
                        "model": model_key,
                        "mode": mode,
                        "group_type": "supercategory",
                        "group": supercategory,
                        **summary,
                    }
                )
            for subject, summary in mode_results["by_subject"].items():
                rows.append({"model": model_key, "mode": mode, "group_type": "subject", "group": subject, **summary})
    return rows


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def write_markdown_report(path: Path, evaluation: dict[str, Any]) -> None:
    lines = [
        "# GPT-2 MMLU Evaluation",
        "",
        "## Protocol",
        "",
        "- Dataset: `cais/mmlu`.",
        "- Test split is scored; dev split supplies up to five few-shot examples.",
        "- Prediction is the largest normalized next-token probability among `A`, `B`, `C`, and `D`.",
        "- The local `gpt2` baseline is the small GPT-2 checkpoint, not the 1.5B GPT-2 model cited in the MMLU appendix.",
    ]
    observed_examples = evaluation["dataset"].get("test_examples")
    paper_examples = evaluation["dataset"].get("paper_reported_test_examples")
    if observed_examples and paper_examples and observed_examples != paper_examples:
        lines.append(
            f"- This `cais/mmlu` snapshot exposes {observed_examples} test examples; "
            f"the paper reports {paper_examples}."
        )

    lines.extend(
        [
            "",
            "## Overall Accuracy",
            "",
            "| Model | Mode | Examples | Accuracy (%) | Avg confidence (%) | RMS cal. error (%) | Truncated | Mean shots |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for model_key, model_results in evaluation["results"].items():
        for mode, mode_results in model_results.items():
            summary = mode_results["overall"]
            lines.append(
                f"| {model_key} | {mode} | {summary['n']} | {pct(summary['accuracy'])} | "
                f"{pct(summary['avg_confidence'])} | {pct(summary['rms_calibration_error'])} | "
                f"{summary['truncated_count']} | {summary['n_shot_mean']:.2f} |"
            )

    lines.extend(["", "## Supercategories", "", "| Model | Mode | Supercategory | Examples | Accuracy (%) | Avg confidence (%) |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for model_key, model_results in evaluation["results"].items():
        for mode, mode_results in model_results.items():
            for supercategory, summary in mode_results["by_supercategory"].items():
                lines.append(
                    f"| {model_key} | {mode} | {supercategory} | {summary['n']} | "
                    f"{pct(summary['accuracy'])} | {pct(summary['avg_confidence'])} |"
                )

    lines.extend(["", "## Lowest-Accuracy Subjects", ""])
    for model_key, model_results in evaluation["results"].items():
        for mode, mode_results in model_results.items():
            subject_items = sorted(
                mode_results["by_subject"].items(),
                key=lambda item: (item[1]["accuracy"], item[0]),
            )[:10]
            lines.extend([f"### {model_key} / {mode}", "", "| Subject | Supercategory | Examples | Accuracy (%) |", "| --- | --- | ---: | ---: |"])
            for subject, summary in subject_items:
                lines.append(
                    f"| {subject_label(subject)} | {SUPER_CATEGORIES[subject]} | {summary['n']} | {pct(summary['accuracy'])} |"
                )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, evaluation: dict[str, Any], all_rows: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    write_json(output_dir / "evaluation.json", evaluation)

    for (model_key, mode), rows in sorted(all_rows.items()):
        write_jsonl(output_dir / f"predictions_{model_key}_{mode}.jsonl", rows)

    metric_fields = [
        "model",
        "mode",
        "subject",
        "subject_label",
        "supercategory",
        "n",
        "correct",
        "accuracy",
        "avg_confidence",
        "calibration_gap",
        "expected_calibration_error",
        "rms_calibration_error",
        "n_shot_mean",
        "n_shot_min",
        "n_shot_max",
        "truncated_count",
    ]
    write_csv(output_dir / "subject_results.csv", flatten_subject_rows(evaluation), metric_fields)

    supercategory_fields = [field for field in metric_fields if field not in {"subject", "subject_label", "supercategory"}]
    supercategory_fields.insert(2, "supercategory")
    write_csv(output_dir / "supercategory_results.csv", flatten_supercategory_rows(evaluation), supercategory_fields)

    calibration_fields = [
        "model",
        "mode",
        "group_type",
        "group",
        "n",
        "correct",
        "accuracy",
        "avg_confidence",
        "calibration_gap",
        "expected_calibration_error",
        "rms_calibration_error",
        "n_shot_mean",
        "n_shot_min",
        "n_shot_max",
        "truncated_count",
    ]
    write_csv(output_dir / "calibration_results.csv", flatten_calibration_rows(evaluation), calibration_fields)
    write_markdown_report(output_dir / "analysis_report.md", evaluation)


def count_test_examples(all_rows: dict[tuple[str, str], list[dict[str, Any]]]) -> int:
    first_rows = next(iter(all_rows.values()), [])
    return len(first_rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    models = parse_csv_arg(args.models)
    modes = parse_csv_arg(args.modes)
    if not models:
        raise ValueError("At least one model must be selected with --models.")
    if not modes:
        raise ValueError("At least one mode must be selected with --modes.")
    validate_choices(models, {"base", "sft"}, "models")
    validate_choices(modes, {"zero_shot", "five_shot"}, "modes")

    model_paths = {"base": args.base_model, "sft": args.sft_model}
    subjects = resolve_subjects(args)
    output_dir = Path(args.output_dir)

    print(f"Loading MMLU subjects: {len(subjects)}", flush=True)
    subject_rows = load_subject_rows(args, subjects)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for model_key in models:
        model_path = model_paths[model_key]
        print(f"Loading model {model_key}: {model_path}", flush=True)
        tokenizer = prepare_tokenizer(model_path)
        model = load_model(model_path, device)
        for mode in modes:
            print(f"Preparing prompts for {model_key}/{mode}", flush=True)
            prepared = prepare_examples(mode, subject_rows, tokenizer, args.max_length)
            print(f"Scoring {len(prepared)} examples for {model_key}/{mode}", flush=True)
            rows = evaluate_prepared_examples(
                model_key,
                model_path,
                mode,
                prepared,
                tokenizer,
                model,
                device,
                args.batch_size,
            )
            all_rows[(model_key, mode)] = rows
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    evaluation = build_evaluation(all_rows, args, subjects)
    evaluation["dataset"]["test_examples"] = count_test_examples(all_rows)
    write_outputs(output_dir, evaluation, all_rows)

    print(f"Saved MMLU evaluation to {output_dir / 'evaluation.json'}", flush=True)
    print(f"Saved report to {output_dir / 'analysis_report.md'}", flush=True)


if __name__ == "__main__":
    main()
