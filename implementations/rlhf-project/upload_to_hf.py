#!/usr/bin/env python3
"""Upload the Dolly RLHF datasets to a public Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi


CONFIGS = ("sft", "rm_schema", "rm_synthetic", "ppo")
SPLITS = ("train", "validation", "test")
DEFAULT_REPO_NAME = "dolly-15k-rlhf-instructgpt-format"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package and upload Dolly RLHF datasets to Hugging Face Hub."
    )
    parser.add_argument("--dataset-dir", default="rlhf_dolly_datasets")
    parser.add_argument("--staging-dir", default="hf_upload")
    parser.add_argument("--repo-id", default=None, help="Full repo id, e.g. username/repo-name.")
    parser.add_argument("--repo-name", default=DEFAULT_REPO_NAME)
    parser.add_argument("--env-file", default=None, help="Optional explicit .env path.")
    return parser.parse_args()


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value

    return values


def get_hf_token(explicit_env_file: str | None) -> str:
    candidate_paths = []
    if explicit_env_file:
        candidate_paths.append(Path(explicit_env_file))
    candidate_paths.extend([Path(".env"), Path("..") / ".env"])

    for path in candidate_paths:
        token = parse_env_file(path).get("HF_TOKEN")
        if token:
            return token

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    raise RuntimeError("HF_TOKEN was not found in .env or the environment.")


def resolve_repo_id(api: HfApi, token: str, repo_id: str | None, repo_name: str) -> str:
    if repo_id:
        return repo_id

    whoami = api.whoami(token=token)
    namespace = whoami.get("name")
    if not namespace:
        raise RuntimeError("Could not infer Hugging Face namespace from HF_TOKEN.")

    return f"{namespace}/{repo_name}"


def write_dataset_card(staging_dir: Path, repo_id: str, counts: dict[str, dict[str, int]]) -> None:
    counts_lines = []
    for config in CONFIGS:
        split_counts = ", ".join(f"{split}={counts[config][split]}" for split in SPLITS)
        counts_lines.append(f"- `{config}`: {split_counts}")

    config_yaml = "\n".join(
        [
            f"- config_name: {config}\n"
            f"  data_files:\n"
            f"  - split: train\n"
            f"    path: data/{config}/train.parquet\n"
            f"  - split: validation\n"
            f"    path: data/{config}/validation.parquet\n"
            f"  - split: test\n"
            f"    path: data/{config}/test.parquet"
            for config in CONFIGS
        ]
    )

    card = f"""---
license: cc-by-sa-3.0
task_categories:
- text-generation
- question-answering
- summarization
language:
- en
size_categories:
- 10K<n<100K
pretty_name: Dolly 15k RLHF Datasets in InstructGPT Format
tags:
- rlhf
- sft
- reward-modeling
- ppo
- instructgpt
- databricks-dolly-15k
configs:
{config_yaml}
---

# Dolly 15k RLHF Datasets in InstructGPT Format

This repository packages `databricks/databricks-dolly-15k` into three RLHF-oriented
dataset configurations inspired by the InstructGPT data flow:

- `sft`: supervised fine-tuning examples with `prompt`, `completion`, and `text`.
- `rm_schema`: reward-modeling schema/prompt pool with empty `chosen` and `rejected`
  fields, `reference_response`, and `ready_for_rm=false`.
- `rm_synthetic`: reward-modeling proxy pairs where Dolly `reference_response` is
  used as `chosen` and sampled GPT-2 SFT output is used as `rejected`.
- `ppo`: prompt-only examples for PPO/RLHF rollouts.

`rm_synthetic` is useful for exercising a reward-modeling pipeline, but it is not
human preference data.

## Format

Prompts use a plain textual InstructGPT-style format:

```text
{{instruction}}

Context:
{{context}}
```

Rows without context use only the instruction text.

## Splits

{chr(10).join(counts_lines)}

## Usage

```python
from datasets import load_dataset

sft = load_dataset("{repo_id}", "sft")
rm_schema = load_dataset("{repo_id}", "rm_schema")
rm_synthetic = load_dataset("{repo_id}", "rm_synthetic")
ppo = load_dataset("{repo_id}", "ppo")
```

## Source and License

Derived from `databricks/databricks-dolly-15k`, released under CC BY-SA 3.0.
Source row ids are preserved in `source_id`.
"""
    (staging_dir / "README.md").write_text(card, encoding="utf-8")


def build_staging_dir(dataset_dir: Path, staging_dir: Path, repo_id: str) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    data_dir = staging_dir / "data"
    counts: dict[str, dict[str, int]] = {}

    for config in CONFIGS:
        dataset_dict = load_from_disk(str(dataset_dir / config))
        counts[config] = {}
        for split in SPLITS:
            split_dir = data_dir / config
            split_dir.mkdir(parents=True, exist_ok=True)
            output_path = split_dir / f"{split}.parquet"
            dataset = dataset_dict[split]
            dataset.to_parquet(str(output_path))
            counts[config][split] = len(dataset)

    write_dataset_card(staging_dir, repo_id, counts)
    shutil.copy2("build_dolly_rlhf_datasets.py", staging_dir / "build_dolly_rlhf_datasets.py")
    shutil.copy2("requirements-rlhf.txt", staging_dir / "requirements-rlhf.txt")


def main() -> None:
    args = parse_args()
    api = HfApi()
    token = get_hf_token(args.env_file)
    repo_id = resolve_repo_id(api, token, args.repo_id, args.repo_name)

    dataset_dir = Path(args.dataset_dir)
    staging_dir = Path(args.staging_dir)
    build_staging_dir(dataset_dir, staging_dir, repo_id)

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True, token=token)
    api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=False, token=token)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staging_dir,
        token=token,
        commit_message="Upload Dolly 15k RLHF datasets",
    )

    print(f"Uploaded public dataset repo: https://huggingface.co/datasets/{repo_id}")
    print(f"Commit: {commit.oid}")


if __name__ == "__main__":
    main()
