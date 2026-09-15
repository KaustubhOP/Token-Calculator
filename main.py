#!/usr/bin/env python3
"""
Token Counter — Project #1 in the RAG learning ladder.

Reads text file(s) and reports character, word and token counts,
plus a rough API cost estimate.

Usage:
    python token_counter.py notes.txt
    python token_counter.py ./docs --model gemini-2.0-flash
    python token_counter.py notes.txt --peek 25
"""

import argparse
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    sys.exit("tiktoken is not installed. Run: pip install tiktoken")


# Rough public list prices in USD per 1 million tokens.
# Input/output prices differ; we use input price since that's what
# your documents would be charged as in a RAG pipeline.
MODEL_PRICES = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "claude-sonnet-4-6": 3.00,
    "claude-haiku-4-5": 1.00,
    "gemini-2.0-flash": 0.10,
}

TEXT_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".html"}


def get_encoder():
    """cl100k_base is the tokenizer used by GPT-4 / GPT-3.5.

    Claude and Gemini tokenize slightly differently, so treat the
    numbers here as a good ballpark rather than an exact count.
    """
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        sys.exit(
            "Could not load the tokenizer vocabulary.\n"
            "tiktoken downloads it on first run, so you need internet access\n"
            "the very first time (it's cached afterwards).\n"
            f"Underlying error: {exc}"
        )


def collect_files(target: Path) -> list[Path]:
    """Return a list of text files to process."""
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(
            p for p in target.rglob("*")
            if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
        )
        if not files:
            sys.exit(f"No text files found in {target}")
        return files
    sys.exit(f"Path does not exist: {target}")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"  ! Skipping {path.name} — not valid UTF-8 text")
        return ""
    except PermissionError:
        print(f"  ! Skipping {path.name} — permission denied")
        return ""


def analyse(text: str, encoder) -> dict:
    tokens = encoder.encode(text)
    words = len(text.split())
    return {
        "chars": len(text),
        "words": words,
        "tokens": len(tokens),
        "ratio": len(tokens) / words if words else 0.0,
        "token_ids": tokens,
    }


def estimate_cost(token_count: int, model: str) -> float:
    price_per_million = MODEL_PRICES[model]
    return token_count / 1_000_000 * price_per_million


def show_token_split(token_ids: list[int], encoder, limit: int) -> None:
    """Decode tokens one at a time so you can SEE how text gets split.

    This is the part that makes tokenization click notice how common
    words are one token, but rare words and names get chopped up.
    """
    print(f"\n  First {limit} tokens, decoded individually:")
    pieces = []
    for tid in token_ids[:limit]:
        piece = encoder.decode([tid])
        # Make whitespace visible
        pieces.append(repr(piece))
    print("  " + " | ".join(pieces))


def report(path: Path, stats: dict, model: str) -> None:
    cost = estimate_cost(stats["tokens"], model)
    print(f"\n{path.name}")
    print(f"  Characters : {stats['chars']:,}")
    print(f"  Words      : {stats['words']:,}")
    print(f"  Tokens     : {stats['tokens']:,}")
    print(f"  Tokens/word: {stats['ratio']:.2f}")
    print(f"  Est. cost  : ${cost:.6f}  ({model}, as input)")


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in a text file or folder of text files."
    )
    parser.add_argument("path", help="Path to a text file or a folder")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=sorted(MODEL_PRICES),
        help="Model to price against (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=0,
        metavar="N",
        help="Show the first N tokens decoded individually",
    )
    args = parser.parse_args()

    encoder = get_encoder()
    files = collect_files(Path(args.path))

    total_tokens = 0
    total_words = 0
    processed = 0

    for file_path in files:
        text = read_text(file_path)
        if not text.strip():
            continue
        stats = analyse(text, encoder)
        report(file_path, stats, args.model)
        if args.peek:
            show_token_split(stats["token_ids"], encoder, args.peek)
        total_tokens += stats["tokens"]
        total_words += stats["words"]
        processed += 1

    if processed > 1:
        ratio = total_tokens / total_words if total_words else 0
        print("\n" + "-" * 40)
        print(f"TOTAL across {processed} files")
        print(f"  Words      : {total_words:,}")
        print(f"  Tokens     : {total_tokens:,}")
        print(f"  Tokens/word: {ratio:.2f}")
        print(f"  Est. cost  : ${estimate_cost(total_tokens, args.model):.6f}")


if __name__ == "__main__":
    main()
    