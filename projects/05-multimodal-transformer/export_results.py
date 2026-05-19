#!/usr/bin/env python3
"""Export Markdown and LaTeX summaries for a completed evaluation run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing responses.jsonl and optionally votes.csv")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_votes(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rel(path: str, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except Exception:
        return path


def write_comparison(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)

    lines: list[str] = ["# Side-by-side Comparison", ""]
    for prompt_id in sorted(by_prompt):
        items = by_prompt[prompt_id]
        first = items[0]
        lines.extend(
            [
                f"## {prompt_id} ({first.get('category', '')})",
                "",
                first.get("prompt", "").strip(),
                "",
            ]
        )
        if first.get("images"):
            lines.append("Images:")
            for image in first["images"]:
                lines.append(f"- `{image.get('id')}`: `{rel(image.get('path', ''), run_dir)}`")
            lines.append("")

        for row in items:
            lines.extend(
                [
                    f"### {row.get('provider')} / {row.get('model_id')}",
                    "",
                    row.get("response_text") or f"ERROR: {row.get('error', '')}",
                    "",
                ]
            )
            if row.get("rendered_images"):
                lines.append("Rendered images:")
                for image in row["rendered_images"]:
                    if "path" in image:
                        lines.append(f"- `{image['path']}` from caption: {image.get('caption', '')}")
                    else:
                        lines.append(f"- render error for caption: {image.get('caption', '')} ({image.get('error', '')})")
                lines.append("")

    (run_dir / "comparison.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize_votes(votes: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(vote["winner_model"] for vote in votes)
    raw_votes = Counter(vote["vote"] for vote in votes)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for vote in votes:
        by_category[vote["category"]][vote["winner_model"]] += 1
    return {
        "total_votes": len(votes),
        "winner_counts": dict(counts),
        "raw_votes": dict(raw_votes),
        "by_category": {category: dict(counter) for category, counter in sorted(by_category.items())},
    }


def write_summary(run_dir: Path, rows: list[dict[str, Any]], votes: list[dict[str, str]]) -> None:
    model_counts = Counter(f"{row.get('provider')} / {row.get('model_id')}" for row in rows)
    errors = [row for row in rows if row.get("error")]
    vote_summary = summarize_votes(votes)

    lines = [
        "# Evaluation Summary",
        "",
        f"- Responses: {len(rows)}",
        f"- Prompts: {len(set(row['prompt_id'] for row in rows))}",
        f"- Votes: {vote_summary['total_votes']}",
        "",
        "## Models",
        "",
    ]
    for model, count in sorted(model_counts.items()):
        lines.append(f"- {model}: {count} response(s)")

    lines.extend(["", "## Vote Counts", ""])
    if votes:
        for winner, count in sorted(vote_summary["winner_counts"].items()):
            lines.append(f"- {winner}: {count}")
    else:
        lines.append("- No votes recorded yet.")

    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors:
            lines.append(f"- {row.get('prompt_id')} / {row.get('model_id')}: {row.get('error')}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a small methodological reproduction of the Chameleon human evaluation protocol, not a benchmark-equivalent replication.",
            "- Public Chameleon checkpoints in Transformers are used here for image/text-to-text generation; bitmap image output is represented with `<caption>...</caption>` placeholders.",
        ]
    )

    (run_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(vote_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_table(run_dir: Path, votes: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lll}",
        r"\hline",
        r"Prompt & Categoria & Vencedor \\",
        r"\hline",
    ]
    if votes:
        for vote in votes:
            winner = vote["winner_model"] if vote["winner_model"] != "tie" else "Empate"
            lines.append(
                f"{latex_escape(vote['prompt_id'])} & {latex_escape(vote['category'])} & {latex_escape(winner)} \\\\"
            )
    else:
        lines.append(r"\multicolumn{3}{c}{Sem votos registrados} \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Resultado da avaliação humana pareada em escala reduzida, inspirada no protocolo do Chameleon.}",
            r"\label{tab:chameleon-mini-eval}",
            r"\end{table}",
            "",
        ]
    )
    (run_dir / "latex_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    responses = read_jsonl(run_dir / "responses.jsonl")
    votes = read_votes(run_dir / "votes.csv")
    write_comparison(run_dir, responses)
    write_summary(run_dir, responses, votes)
    write_latex_table(run_dir, votes)
    print(f"Wrote comparison.md, summary.md, summary.json, and latex_table.tex in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
