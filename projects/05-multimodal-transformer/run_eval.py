#!/usr/bin/env python3
"""Run a small Chameleon-style human-evaluation reproduction.

The script compares a local Hugging Face Chameleon checkpoint with an OpenAI
vision-capable model over a fixed prompt suite. It writes JSONL incrementally so
partial runs remain useful if a model call fails.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
DEFAULT_PROMPTS = PROJECT_DIR / "prompts.json"
DEFAULT_OUTPUTS = PROJECT_DIR / "outputs"
DEFAULT_IMAGE_CACHE = PROJECT_DIR / "assets" / "input_images"

CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.IGNORECASE | re.DOTALL)

SYSTEM_INSTRUCTION = """You are participating in a small reproduction of the Chameleon paper's human evaluation.
Answer in English. Be useful, specific, and concise enough for side-by-side human judging.
If the prompt asks for images or illustrations, do not use Markdown image syntax and do not claim that you actually created a bitmap image. Instead, put each intended image as a standalone <caption>...</caption> block with a vivid, renderable caption.
"""


@dataclass
class LocalImage:
    image_id: str
    path: Path
    description: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-id", action="append", default=None, help="Run only the selected prompt id. Can be repeated.")
    parser.add_argument("--check-only", action="store_true", help="Validate environment, imports, GPU, and model access, then exit.")

    parser.add_argument("--model", default="facebook/chameleon-30b", help="Primary Chameleon model id.")
    parser.add_argument("--fallback-model", default="facebook/chameleon-7b", help="Fallback Chameleon model id.")
    parser.add_argument("--no-fallback", action="store_true", help="Do not fall back to the fallback Chameleon model.")
    parser.add_argument(
        "--chameleon-quantization",
        choices=["auto", "none", "4bit", "8bit"],
        default="auto",
        help="Quantization for the Chameleon checkpoint. auto uses 4bit for 30B and bf16 for smaller checkpoints.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--skip-chameleon", action="store_true")

    parser.add_argument("--openai-model", default="gpt-5-nano")
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--openai-image-detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--skip-openai", action="store_true")

    parser.add_argument("--openai-image-render", action="store_true", help="Render <caption> blocks through the OpenAI image API.")
    parser.add_argument("--no-openai-image-render", action="store_true", help="Compatibility flag; image rendering is disabled unless --openai-image-render is set.")
    parser.add_argument("--openai-image-model", default="gpt-image-1-mini")
    parser.add_argument("--openai-image-size", default="1024x1024")

    return parser.parse_args()


def load_environment() -> None:
    load_dotenv_file(REPO_ROOT / ".env")
    load_dotenv_file(PROJECT_DIR / ".env")


def load_dotenv_file(path: Path, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or (key in os.environ and not override):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Expected it in {REPO_ROOT / '.env'} or the process environment.")
    return value


def load_prompts(path: Path, limit: int | None, prompt_ids: list[str] | None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if prompt_ids:
        wanted = set(prompt_ids)
        data = [item for item in data if item["id"] in wanted]
        missing = wanted - {item["id"] for item in data}
        if missing:
            raise RuntimeError(f"Prompt ids not found: {', '.join(sorted(missing))}")
    if limit is not None:
        data = data[:limit]
    return data


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(output_root: Path) -> Path:
    out_dir = output_root / run_id()
    out_dir.mkdir(parents=True, exist_ok=False)
    latest = output_root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    try:
        latest.symlink_to(out_dir.name, target_is_directory=True)
    except OSError:
        latest.mkdir(parents=True, exist_ok=True)
    return out_dir


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slug_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def resolve_image(entry: dict[str, Any], image_cache: Path) -> LocalImage:
    image_cache.mkdir(parents=True, exist_ok=True)
    image_id = entry["id"]
    description = entry.get("description", "")
    if entry.get("source") == "local":
        path = (PROJECT_DIR / entry["path"]).resolve()
        if not path.exists():
            raise RuntimeError(f"Local image not found for {image_id}: {path}")
        return LocalImage(image_id=image_id, path=path, description=description)

    url = entry["url"]
    filename = entry.get("filename") or slug_filename(Path(url).name or f"{image_id}.jpg")
    path = image_cache / filename
    if not path.exists():
        response = requests.get(url, timeout=60, headers={"User-Agent": "chameleon-eval/1.0"})
        response.raise_for_status()
        path.write_bytes(response.content)
    return LocalImage(image_id=image_id, path=path.resolve(), description=description)


def collect_images(prompt: dict[str, Any], image_cache: Path) -> list[LocalImage]:
    return [resolve_image(entry, image_cache) for entry in prompt.get("images", [])]


def build_model_prompt(prompt: dict[str, Any], image_count: int, for_chameleon: bool) -> str:
    body = prompt["prompt"].strip()
    if prompt.get("requires_image_output"):
        body += "\n\nWhere an image would appear in the answer, include a standalone <caption>...</caption> block."
    else:
        body += "\n\nDo not add image captions unless the prompt explicitly needs them."
    body = SYSTEM_INSTRUCTION.strip() + "\n\nUser prompt:\n" + body
    if for_chameleon and image_count:
        return ("<image>" * image_count) + "\n" + body
    return body


def to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return str(value)


def check_hf_access(model_id: str, token: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        response = requests.get(url, timeout=30, headers={"Authorization": f"Bearer {token}"})
        return {"model": model_id, "status_code": response.status_code, "ok": response.ok}
    except requests.RequestException as exc:
        return {
            "model": model_id,
            "status_code": None,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_environment(args: argparse.Namespace) -> None:
    print("Project:", PROJECT_DIR)
    print("Python:", sys.version.split()[0])
    hf_token = require_env("HF_TOKEN")
    if not args.skip_openai:
        require_env("OPENAI_API_KEY")

    hf_access = check_hf_access(args.model, hf_token)
    print("HF access:", hf_access)
    if not hf_access["ok"] and hf_access.get("error"):
        print("HF access warning: could not reach Hugging Face. This is usually DNS/network, not an auth failure.")
    if not args.no_fallback and args.fallback_model != args.model:
        hf_fallback_access = check_hf_access(args.fallback_model, hf_token)
        print("HF fallback access:", hf_fallback_access)
        if not hf_fallback_access["ok"] and hf_fallback_access.get("error"):
            print("HF fallback warning: could not reach Hugging Face for the fallback model either.")

    import torch
    import transformers

    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"gpu {i}: {props.name}, {props.total_memory / (1024 ** 3):.1f} GiB")

    if not args.skip_openai:
        from openai import OpenAI

        client = OpenAI()
        print("OpenAI SDK: client initialized")


def chameleon_quantization(model_id: str, setting: str) -> str:
    if setting != "auto":
        return setting
    return "4bit" if "30b" in model_id.lower() else "none"


def load_chameleon(model_id: str, quantization: str, hf_token: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import BitsAndBytesConfig, ChameleonForConditionalGeneration, ChameleonProcessor

    quantization = chameleon_quantization(model_id, quantization)
    kwargs: dict[str, Any] = {
        "token": hf_token,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    processor = ChameleonProcessor.from_pretrained(model_id, token=hf_token)
    model = ChameleonForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    return processor, model, quantization


def first_model_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    return "cuda"


def run_chameleon_one(
    processor: Any,
    model: Any,
    model_id: str,
    quantization: str,
    prompt: dict[str, Any],
    images: list[LocalImage],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    text = build_model_prompt(prompt, image_count=len(images), for_chameleon=True)
    pil_images = [Image.open(image.path).convert("RGB") for image in images]

    processor_kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt"}
    if pil_images:
        processor_kwargs["images"] = pil_images
    inputs = processor(**processor_kwargs)
    inputs = inputs.to(first_model_device(model))

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_len = int(inputs["input_ids"].shape[-1])
    generated_ids = output_ids[0][prompt_len:]
    response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    if not response_text:
        response_text = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    return {
        "provider": "huggingface",
        "model_id": model_id,
        "quantization": quantization,
        "response_text": response_text,
        "latency_s": round(time.perf_counter() - started, 3),
        "usage": {"max_new_tokens": args.max_new_tokens},
    }


def create_openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def run_openai_one(client: Any, prompt: dict[str, Any], images: list[LocalImage], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    text = build_model_prompt(prompt, image_count=len(images), for_chameleon=False)
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for image in images:
        content.append(
            {
                "type": "input_image",
                "image_url": to_data_url(image.path),
                "detail": args.openai_image_detail,
            }
        )

    response = client.responses.create(
        model=args.openai_model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=args.max_output_tokens,
    )

    return {
        "provider": "openai",
        "model_id": args.openai_model,
        "response_text": extract_openai_text(response),
        "latency_s": round(time.perf_counter() - started, 3),
        "usage": jsonable(getattr(response, "usage", None)),
        "response_id": getattr(response, "id", None),
    }


def extract_captions(text: str) -> list[str]:
    return [match.strip() for match in CAPTION_RE.findall(text or "") if match.strip()]


def render_caption_images(
    client: Any,
    row: dict[str, Any],
    run_dir: Path,
    image_model: str,
    image_size: str,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    captions = extract_captions(row.get("response_text", ""))
    if not captions:
        return rendered

    target_dir = run_dir / "rendered_images" / row["provider"] / row["prompt_id"]
    target_dir.mkdir(parents=True, exist_ok=True)

    for idx, caption in enumerate(captions, start=1):
        out_path = target_dir / f"caption_{idx:02d}.png"
        try:
            image_response = client.images.generate(
                model=image_model,
                prompt=caption,
                size=image_size,
            )
            image_b64 = image_response.data[0].b64_json
            out_path.write_bytes(base64.b64decode(image_b64))
            rendered.append({"caption": caption, "path": str(out_path.relative_to(run_dir))})
        except Exception as exc:  # noqa: BLE001 - keep eval run alive.
            rendered.append({"caption": caption, "error": str(exc)})
    return rendered


def error_row(provider: str, model_id: str, exc: BaseException) -> dict[str, Any]:
    return {
        "provider": provider,
        "model_id": model_id,
        "response_text": "",
        "latency_s": None,
        "usage": None,
        "error": str(exc),
        "traceback": traceback.format_exc(limit=4),
    }


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def base_row(run_name: str, prompt: dict[str, Any], images: list[LocalImage]) -> dict[str, Any]:
    return {
        "run_id": run_name,
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "input_modality": prompt["input_modality"],
        "requires_image_output": bool(prompt.get("requires_image_output")),
        "prompt": prompt["prompt"],
        "images": [
            {
                "id": image.image_id,
                "path": str(image.path),
                "description": image.description,
            }
            for image in images
        ],
    }


def write_run_manifest(run_dir: Path, args: argparse.Namespace, prompts: list[dict[str, Any]]) -> None:
    dump_json(
        run_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "args": jsonable(vars(args)),
            "prompt_count": len(prompts),
            "notes": [
                "Chameleon public Transformers checkpoints are image-text-to-text; bitmap image generation is not available in this runner.",
                "Prompts that ask for image output use <caption>...</caption> placeholders for fair side-by-side judging.",
            ],
        },
    )


def main() -> int:
    args = parse_args()
    load_environment()

    if args.check_only:
        check_environment(args)
        return 0

    prompts = load_prompts(args.prompts, args.limit, args.prompt_id)
    run_dir = make_run_dir(args.output_dir)
    responses_path = run_dir / "responses.jsonl"
    write_run_manifest(run_dir, args, prompts)

    hf_token = require_env("HF_TOKEN") if not args.skip_chameleon else os.getenv("HF_TOKEN", "")
    if not args.skip_openai:
        require_env("OPENAI_API_KEY")

    chameleon_loaded: tuple[Any, Any, str, str] | None = None
    if not args.skip_chameleon:
        try:
            processor, model, quantization = load_chameleon(args.model, args.chameleon_quantization, hf_token)
            chameleon_loaded = (processor, model, args.model, quantization)
        except Exception as exc:
            if args.no_fallback or args.fallback_model == args.model:
                raise
            print(f"Primary Chameleon load failed: {exc}", file=sys.stderr)
            print(f"Trying fallback model: {args.fallback_model}", file=sys.stderr)
            cleanup_cuda()
            processor, model, quantization = load_chameleon(args.fallback_model, args.chameleon_quantization, hf_token)
            chameleon_loaded = (processor, model, args.fallback_model, quantization)

    openai_client = None
    if not args.skip_openai or args.openai_image_render:
        openai_client = create_openai_client()

    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] {prompt['id']}")
        images = collect_images(prompt, args.image_cache)
        common = base_row(run_dir.name, prompt, images)

        if chameleon_loaded and not args.skip_chameleon:
            processor, model, model_id, quantization = chameleon_loaded
            try:
                result = run_chameleon_one(processor, model, model_id, quantization, prompt, images, args)
            except Exception as exc:  # noqa: BLE001 - keep eval run alive.
                result = error_row("huggingface", model_id, exc)
            row = {**common, **result}
            if args.openai_image_render and openai_client:
                row["rendered_images"] = render_caption_images(
                    openai_client,
                    row,
                    run_dir,
                    args.openai_image_model,
                    args.openai_image_size,
                )
            append_jsonl(responses_path, row)

        if not args.skip_openai:
            try:
                result = run_openai_one(openai_client, prompt, images, args)
            except Exception as exc:  # noqa: BLE001 - keep eval run alive.
                result = error_row("openai", args.openai_model, exc)
            row = {**common, **result}
            if args.openai_image_render and openai_client:
                row["rendered_images"] = render_caption_images(
                    openai_client,
                    row,
                    run_dir,
                    args.openai_image_model,
                    args.openai_image_size,
                )
            append_jsonl(responses_path, row)

    print(f"Wrote {responses_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
