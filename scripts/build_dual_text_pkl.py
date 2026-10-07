"""Add Qwen-enhanced BERT tensors to a copy of an MMSA-format PKL."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", default="datasets/MOSI/unaligned_50.pkl"
    )
    parser.add_argument(
        "--enhanced-path", default="datasets/MOSI/qwen25_enhanced.jsonl"
    )
    parser.add_argument(
        "--output-path",
        default="datasets/MOSI/unaligned_50_dual_qwen25.pkl",
    )
    parser.add_argument(
        "--bert-path", default="pretrained/bert-base-uncased"
    )
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument(
        "--long-text-policy",
        choices=["fallback", "truncate", "error"],
        default="fallback",
        help="Default avoids silently truncating a generated rewrite.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Use raw text for missing records; otherwise missing IDs are fatal.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing output PKL.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", str(value)).strip()


def load_records(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (str(record["split"]), normalize_text(record["id"]))
                enhanced = normalize_text(record["enhanced_text"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid enhancement record at line {line_number}"
                ) from exc
            if not enhanced:
                raise ValueError(f"Empty enhanced_text at line {line_number}")
            if key in records:
                duplicates += 1
            records[key] = record
    return records, duplicates


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    enhanced_path = Path(args.enhanced_path)
    output_path = Path(args.output_path)
    bert_path = Path(args.bert_path)

    if data_path.resolve() == output_path.resolve():
        raise ValueError("--output-path must not overwrite the original dataset")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite explicitly."
        )
    if not bert_path.is_dir():
        raise FileNotFoundError(f"Local BERT directory not found: {bert_path}")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to build text_bert_llm") from exc

    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    records, duplicate_count = load_records(enhanced_path)
    if duplicate_count:
        raise ValueError(
            f"Enhancement JSONL contains {duplicate_count} duplicate IDs. "
            "Validate or clean it before building the PKL."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(bert_path), local_files_only=True, use_fast=True
    )
    build_stats: dict[str, Any] = {}
    used_keys: set[tuple[str, str]] = set()

    for split in ("train", "valid", "test"):
        raw_texts = [normalize_text(value) for value in data[split]["raw_text"]]
        ids = [normalize_text(value) for value in data[split]["id"]]
        enhanced_texts: list[str] = []
        missing_count = 0
        raw_mismatch_count = 0
        long_generated_count = 0
        long_fallback_count = 0

        for sample_id, raw_text in zip(ids, raw_texts):
            key = (split, sample_id)
            record = records.get(key)
            if record is None:
                missing_count += 1
                if not args.allow_incomplete:
                    continue
                enhanced = raw_text
            else:
                used_keys.add(key)
                record_raw = normalize_text(record.get("raw_text", ""))
                if record_raw != raw_text:
                    raw_mismatch_count += 1
                enhanced = normalize_text(record["enhanced_text"])

            token_length = len(
                tokenizer.encode(enhanced, add_special_tokens=True)
            )
            if token_length > args.max_length:
                long_generated_count += 1
                if args.long_text_policy == "error":
                    raise ValueError(
                        f"{split}:{sample_id} has {token_length} BERT tokens "
                        f"(limit={args.max_length})"
                    )
                if args.long_text_policy == "fallback":
                    enhanced = raw_text
                    long_fallback_count += 1
            enhanced_texts.append(enhanced)

        if missing_count and not args.allow_incomplete:
            raise ValueError(
                f"{split}: missing {missing_count} enhancement records. "
                "Generate all splits or pass --allow-incomplete intentionally."
            )
        if raw_mismatch_count:
            raise ValueError(
                f"{split}: {raw_mismatch_count} JSONL raw texts do not match the PKL"
            )
        if len(enhanced_texts) != len(raw_texts):
            raise RuntimeError(f"{split}: failed to construct all enhanced texts")

        encoded = tokenizer(
            enhanced_texts,
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_tensors="np",
        )
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(encoded["input_ids"])
        text_bert_llm = np.stack(
            [
                encoded["input_ids"],
                encoded["attention_mask"],
                token_type_ids,
            ],
            axis=1,
        ).astype(np.float32)

        if text_bert_llm.shape != data[split]["text_bert"].shape:
            raise ValueError(
                f"{split}: enhanced tensor shape {text_bert_llm.shape} does not "
                f"match original {data[split]['text_bert'].shape}"
            )
        data[split]["raw_text_llm"] = np.asarray(enhanced_texts)
        data[split]["text_bert_llm"] = text_bert_llm
        build_stats[split] = {
            "samples": len(ids),
            "missing_fallbacks": missing_count,
            "over_length_generated": long_generated_count,
            "over_length_fallbacks": long_fallback_count,
            "text_bert_llm_shape": list(text_bert_llm.shape),
        }

    unused_records = len(set(records) - used_keys)
    if unused_records:
        raise ValueError(
            f"Enhancement JSONL has {unused_records} records not present in the PKL"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)

    first_record = next(iter(records.values()), {})
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(data_path),
        "enhancement_jsonl": str(enhanced_path),
        "output_dataset": str(output_path),
        "bert_path": str(bert_path),
        "max_length": args.max_length,
        "long_text_policy": args.long_text_policy,
        "prompt_version": first_record.get("prompt_version"),
        "prompt_sha256": first_record.get("prompt_sha256"),
        "model_path": first_record.get("model_path"),
        "model_config_sha256": first_record.get("model_config_sha256"),
        "stats": build_stats,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
