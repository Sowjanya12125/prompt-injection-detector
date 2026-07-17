"""
collect.py - Dataset builder for prompt injection detector.
Loads raw injection + benign examples, cleans, deduplicates,
and saves a labeled CSV ready for training.

Usage:
    python src/collect.py
"""

import json
import csv
import os
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INJECTIONS_FILE = RAW_DIR / "injections.json"
INJECTIONS_V2_FILE = RAW_DIR / "injections_v2.json"
BENIGN_FILE = RAW_DIR / "benign.json"
OUTPUT_CSV = PROCESSED_DIR / "dataset.csv"


# ── Text cleaning ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove control characters."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)          # collapse multiple spaces
    text = re.sub(r"[^\x20-\x7E]", "", text)  # remove non-ASCII control chars
    return text


# ── Loader ──────────────────────────────────────────────────────────────────
def load_json(filepath: Path) -> list[dict]:
    """Load a JSON array from a file. Returns empty list if file missing."""
    if not filepath.exists():
        print(f"[WARN] File not found: {filepath} — skipping.")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} examples from {filepath}")
    return data


# ── Builder ─────────────────────────────────────────────────────────────────
def build_dataset() -> list[dict]:
    """Merge injections + benign, clean text, deduplicate."""
    injections = load_json(INJECTIONS_FILE)
    injections_v2 = load_json(INJECTIONS_V2_FILE)
    benign = load_json(BENIGN_FILE)

    combined = injections + injections_v2 + benign
    print(f"[INFO] Total raw examples: {len(combined)}")

    seen = set()
    cleaned = []
    duplicates = 0

    for item in combined:
        text = clean_text(item.get("text", ""))
        label = item.get("label", "unknown")

        if not text:
            continue  # skip empty

        if text in seen:
            duplicates += 1
            continue  # skip duplicates

        seen.add(text)
        cleaned.append({
            "text": text,
            "label": label,
            "category": item.get("category", "unknown")
        })

    print(f"[INFO] Removed {duplicates} duplicate(s)")
    print(f"[INFO] Final dataset size: {len(cleaned)}")
    return cleaned


# ── Stats ────────────────────────────────────────────────────────────────────
def print_stats(dataset: list[dict]) -> None:
    """Print class distribution."""
    from collections import Counter
    label_counts = Counter(d["label"] for d in dataset)
    category_counts = Counter(d["category"] for d in dataset)

    print("\n── Label Distribution ──────────────────")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<12} {count}")

    print("\n── Category Distribution ───────────────")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25} {count}")
    print()


# ── Writer ───────────────────────────────────────────────────────────────────
def save_csv(dataset: list[dict], output_path: Path) -> None:
    """Write dataset to CSV."""
    fieldnames = ["text", "label", "category"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"[INFO] Saved dataset to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print(" Prompt Injection Dataset Builder")
    print("=" * 45)

    dataset = build_dataset()
    print_stats(dataset)
    save_csv(dataset, OUTPUT_CSV)

    print("[DONE] Run this again after adding more examples to data/raw/")
