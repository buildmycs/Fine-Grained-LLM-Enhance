"""Validate coverage and basic quality of generated enhancement JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", default="datasets/MOSI/unaligned_50.pkl"
    )
    parser.add_argument(
        "--enhanced-path", default="datasets/MOSI/qwen25_enhanced.jsonl"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--bert-path",
        default=None,
        help="Optional local BERT tokenizer path for exact 50-token checks.",
    )
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", str(value)).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> int:
    args = parse_args()
    with Path(args.data_path).open("rb") as handle:
        data = pickle.load(handle)

    expected: dict[tuple[str, str], str] = {}
    all_known: set[tuple[str, str]] = set()
    for split in ("train", "valid", "test"):
        for sample_id, raw_text in zip(data[split]["id"], data[split]["raw_text"]):
            key = (split, normalize_text(sample_id))
            all_known.add(key)
            if split in args.splits:
                expected[key] = normalize_text(raw_text)

    records: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = 0
    malformed_lines = 0
    unknown_records = 0
    with Path(args.enhanced_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (str(record["split"]), normalize_text(record["id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                malformed_lines += 1
                continue
            if key not in all_known:
                unknown_records += 1
                continue
            if key in records:
                duplicates += 1
            records[key] = record

    selected_records = {
        key: value for key, value in records.items() if key in expected
    }
    missing = sorted(set(expected) - set(selected_records))
    raw_mismatches = 0
    hash_mismatches = 0
    empty_enhancements = 0
    unchanged = 0
    statuses: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    word_lengths: list[int] = []
    texts_for_token_check: list[str] = []

    for key, record in selected_records.items():
        raw_text = expected[key]
        record_raw = normalize_text(record.get("raw_text", ""))
        enhanced = normalize_text(record.get("enhanced_text", ""))
        if record_raw != raw_text:
            raw_mismatches += 1
        if record.get("source_text_sha256") not in (None, sha256_text(raw_text)):
            hash_mismatches += 1
        if not enhanced:
            empty_enhancements += 1
        if enhanced.casefold() == raw_text.casefold():
            unchanged += 1
        statuses[str(record.get("status", "<missing>"))] += 1
        for field in (
            "coreference_completed",
            "ambiguity_resolved",
            "implicit_meaning_explicit",
            "uncertain",
        ):
            if record.get(field) is True:
                operations[field] += 1
        word_lengths.append(len(enhanced.split()))
        texts_for_token_check.append(enhanced)

    over_max_tokens = None
    token_p95 = None
    if args.bert_path:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required when --bert-path is supplied"
            ) from exc
        tokenizer_path = Path(args.bert_path)
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"BERT tokenizer not found: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, use_fast=True
        )
        token_lengths = [
            len(tokenizer.encode(text, add_special_tokens=True))
            for text in texts_for_token_check
        ]
        over_max_tokens = sum(length > args.max_length for length in token_lengths)
        token_p95 = percentile(token_lengths, 0.95)

    count = len(selected_records)
    report = {
        "requested_splits": args.splits,
        "expected_records": len(expected),
        "available_records": count,
        "missing_records": len(missing),
        "missing_examples": [f"{split}:{sample_id}" for split, sample_id in missing[:10]],
        "duplicate_records": duplicates,
        "malformed_jsonl_lines": malformed_lines,
        "unknown_records": unknown_records,
        "raw_text_mismatches": raw_mismatches,
        "source_hash_mismatches": hash_mismatches,
        "empty_enhancements": empty_enhancements,
        "unchanged_records": unchanged,
        "unchanged_rate": round(unchanged / count, 6) if count else None,
        "status_counts": dict(statuses),
        "operation_counts": dict(operations),
        "word_length_mean": round(statistics.mean(word_lengths), 3)
        if word_lengths
        else None,
        "word_length_p95": round(percentile(word_lengths, 0.95), 3)
        if word_lengths
        else None,
        "over_max_bert_tokens": over_max_tokens,
        "bert_token_length_p95": token_p95,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")

    fatal_count = (
        len(missing)
        + malformed_lines
        + unknown_records
        + raw_mismatches
        + hash_mismatches
        + empty_enhancements
    )
    return 1 if fatal_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
