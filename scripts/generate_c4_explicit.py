"""Standalone C4-Explicit Qwen2.5 enhancement generator.

Uses four preceding and two following same-video utterances. Runs with
Transformers + PyTorch SDPA and has no dependency on the other generation
scripts, vLLM, FlashInfer, or CUTLASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "qwen25-c4-explicit-v2"
CONTEXT_BEFORE = 4
CONTEXT_AFTER = 2
OPERATIONS = (
    "coreference_completed",
    "ambiguity_resolved",
    "implicit_meaning_explicit",
)

SYSTEM_PROMPT = """You are an evidence-grounded semantic explicitation system for spoken English transcripts.

Rewrite only TARGET into a concise, self-contained utterance that can be understood without reading the context.

Actively perform a clarification whenever the supplied transcript reliably supports at least one operation:
1. replace or supplement a pronoun with its referent;
2. name the person, object, event, movie, product, or aspect being evaluated;
3. resolve an elliptical, incomplete, vague, or ambiguous spoken expression;
4. make a strongly implied proposition explicit.

Strict constraints:
- Every added semantic element must be supported by a short exact phrase copied from TARGET or CONTEXT.
- Preserve the original sentiment polarity and sentiment intensity exactly.
- Do not add sentiment words merely to make sentiment easier to classify.
- Do not invent entities, events, causes, intentions, opinions, or background knowledge.
- Do not perform stylistic paraphrasing when no semantic clarification is supported.
- If evidence is insufficient, keep TARGET unchanged and explain why in unchanged_reason.
- Remove disfluencies only when doing so does not alter meaning.
- Rewrite TARGET only; never merge multiple utterances into one review.
- Never output a sentiment label or sentiment score.
- Return exactly one valid JSON object with no Markdown or commentary."""

USER_TEMPLATE = """Produce a C4-Explicit rewrite of TARGET.

PRECEDING CONTEXT (oldest to newest):
{before}

TARGET:
{target}

FOLLOWING CONTEXT (nearest to farthest):
{after}

Return exactly:
{{
  "enhanced_text": "a concise self-contained rewrite of TARGET",
  "coreference_completed": true,
  "ambiguity_resolved": false,
  "implicit_meaning_explicit": true,
  "supporting_evidence": ["short exact phrase from TARGET or CONTEXT"],
  "uncertain": false,
  "unchanged_reason": ""
}}

Set an operation flag to true only when that operation changed enhanced_text.
If all operation flags are false, enhanced_text must equal TARGET and
unchanged_reason must be non-empty."""

RETRY_SUFFIX = """

The previous response was invalid. Return one JSON object only. Ensure changed
text has at least one true operation and an exact supporting-evidence phrase.
Unchanged text must have all operation flags false and unchanged_reason set."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", default="datasets/mosi/unaligned_50.pkl"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--output-path",
        default="datasets/mosi/qwen25_c4_explicit.jsonl",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", str(value)).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segment_key(sample_id: str) -> tuple[str, int | None]:
    if "$_$" not in sample_id:
        return sample_id, None
    video_id, segment = sample_id.rsplit("$_$", 1)
    try:
        return video_id, int(segment)
    except ValueError:
        return video_id, None


def load_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    for split in ("train", "valid", "test"):
        if split not in data:
            raise KeyError(f"Dataset has no {split!r} split")
        for field in ("id", "raw_text"):
            if field not in data[split]:
                raise KeyError(f"{split} has no {field!r} field")
    return data


def build_samples(
    data: dict[str, Any], splits: Iterable[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split in splits:
        ids = [normalize(item) for item in data[split]["id"]]
        texts = [normalize(item) for item in data[split]["raw_text"]]
        if len(ids) != len(texts):
            raise ValueError(f"{split}: id/raw_text length mismatch")

        groups: dict[str, list[tuple[int | None, int]]] = defaultdict(list)
        for index, sample_id in enumerate(ids):
            video_id, number = segment_key(sample_id)
            groups[video_id].append((number, index))

        context_indices: dict[int, tuple[list[int], list[int]]] = {}
        for members in groups.values():
            members.sort(
                key=lambda item: (
                    item[0] is None,
                    item[0] if item[0] is not None else item[1],
                    item[1],
                )
            )
            ordered = [item[1] for item in members]
            for position, index in enumerate(ordered):
                before = ordered[max(0, position - CONTEXT_BEFORE) : position]
                after = ordered[position + 1 : position + 1 + CONTEXT_AFTER]
                context_indices[index] = before, after

        for index, (sample_id, raw_text) in enumerate(zip(ids, texts)):
            before, after = context_indices[index]
            result.append(
                {
                    "split": split,
                    "index": index,
                    "id": sample_id,
                    "raw_text": raw_text,
                    "context_before_ids": [ids[i] for i in before],
                    "context_before": [texts[i] for i in before],
                    "context_after_ids": [ids[i] for i in after],
                    "context_after": [texts[i] for i in after],
                }
            )
    return result


def format_before(texts: list[str]) -> str:
    if not texts:
        return "<NONE>"
    count = len(texts)
    return "\n".join(
        f"[BEFORE -{count - index}] {text}"
        for index, text in enumerate(texts)
    )


def format_after(texts: list[str]) -> str:
    if not texts:
        return "<NONE>"
    return "\n".join(
        f"[AFTER +{index}] {text}" for index, text in enumerate(texts, start=1)
    )


def conversation(
    sample: dict[str, Any], retry: bool = False
) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(
        before=format_before(sample["context_before"]),
        target=sample["raw_text"],
        after=format_after(sample["context_after"]),
    )
    if retry:
        user += RETRY_SUFFIX
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_object(response: str) -> dict[str, Any]:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start >= 0:
        try:
            value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("response does not contain a valid JSON object")


def boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field!r} must be a JSON boolean")


def parse_response(response: str, sample: dict[str, Any]) -> dict[str, Any]:
    value = extract_object(response)
    enhanced = normalize(value.get("enhanced_text", ""))
    if not enhanced:
        raise ValueError("enhanced_text is empty")
    if len(enhanced) > 1000:
        raise ValueError("enhanced_text is unexpectedly long")

    parsed: dict[str, Any] = {"enhanced_text": enhanced}
    for field in OPERATIONS:
        parsed[field] = boolean(value.get(field), field)
    parsed["uncertain"] = boolean(value.get("uncertain"), "uncertain")

    evidence = value.get("supporting_evidence")
    if not isinstance(evidence, list) or any(
        not isinstance(item, str) for item in evidence
    ):
        raise ValueError("supporting_evidence must be a JSON string array")
    parsed["supporting_evidence"] = [
        normalize(item) for item in evidence if normalize(item)
    ]
    parsed["unchanged_reason"] = normalize(value.get("unchanged_reason", ""))

    changed = enhanced.casefold() != sample["raw_text"].casefold()
    operated = any(parsed[field] for field in OPERATIONS)
    if changed and not operated:
        raise ValueError("changed text must set an operation to true")
    if changed and not parsed["supporting_evidence"]:
        raise ValueError("changed text must include supporting evidence")
    if not changed and operated:
        raise ValueError("unchanged text must have all operation flags false")
    if not changed and not parsed["unchanged_reason"]:
        raise ValueError("unchanged text must include unchanged_reason")

    transcript = " ".join(
        sample["context_before"]
        + [sample["raw_text"]]
        + sample["context_after"]
    ).casefold()
    unsupported = [
        item
        for item in parsed["supporting_evidence"]
        if item.casefold() not in transcript
    ]
    if unsupported:
        raise ValueError(
            "supporting_evidence is not an exact transcript phrase: "
            + repr(unsupported[:2])
        )
    return parsed


def load_completed(path: Path) -> tuple[set[tuple[str, str]], int]:
    completed: set[tuple[str, str]] = set()
    malformed = 0
    if not path.is_file():
        return completed, malformed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                completed.add((str(item["split"]), normalize(item["id"])))
            except (json.JSONDecodeError, KeyError, TypeError):
                malformed += 1
    return completed, malformed


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch-size must be greater than zero")
    for start in range(0, len(values), size):
        yield values[start : start + size]


class Generator:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "torch, transformers, and accelerate are required"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required")

        self.torch = torch
        self.max_input_tokens = args.max_input_tokens
        self.max_tokens = args.max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
            "dtype": dtype_map[args.dtype],
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path), **kwargs
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path), **kwargs
            )
        self.model.eval()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("Loaded C4-Explicit with Transformers + PyTorch SDPA.")

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            )
            for item in conversations
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(
            output[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def record(
    sample: dict[str, Any],
    parsed: dict[str, Any] | None,
    raw_response: str,
    error: str | None,
    retry_count: int,
    args: argparse.Namespace,
    model_hash: str | None,
) -> dict[str, Any]:
    if parsed is None:
        parsed = {
            "enhanced_text": sample["raw_text"],
            "coreference_completed": False,
            "ambiguity_resolved": False,
            "implicit_meaning_explicit": False,
            "supporting_evidence": [],
            "uncertain": True,
            "unchanged_reason": "Parser failure; raw-text fallback.",
        }
        status = "fallback_parse_error"
    else:
        status = "success_after_retry" if retry_count else "success"
    return {
        "split": sample["split"],
        "index": sample["index"],
        "id": sample["id"],
        "raw_text": sample["raw_text"],
        **parsed,
        "context_before_ids": sample["context_before_ids"],
        "context_after_ids": sample["context_after_ids"],
        "context_before_size": CONTEXT_BEFORE,
        "context_after_size": CONTEXT_AFTER,
        "status": status,
        "error": error,
        "retry_count": retry_count,
        "raw_response": raw_response,
        "source_text_sha256": text_hash(sample["raw_text"]),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": text_hash(SYSTEM_PROMPT + "\n" + USER_TEMPLATE),
        "model_path": args.model_path,
        "model_config_sha256": model_hash,
        "generation": {
            "backend": "transformers",
            "attention": "sdpa",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    if args.max_input_tokens <= args.max_tokens:
        raise ValueError("max-input-tokens must exceed max-tokens")
    data = load_data(Path(args.data_path))
    samples = build_samples(data, args.splits)

    if args.dry_run:
        count = min(args.limit or 3, len(samples))
        print(f"C4-Explicit dry run: {len(samples)} samples; showing {count}")
        for sample in samples[:count]:
            print("=" * 80)
            print(f"{sample['split']} / {sample['id']}")
            print(json.dumps(conversation(sample), ensure_ascii=False, indent=2))
        return 0

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if args.overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed, malformed = load_completed(output_path)
    if malformed:
        print(f"Warning: ignored {malformed} malformed JSONL lines", file=sys.stderr)
    pending = [
        sample
        for sample in samples
        if (sample["split"], sample["id"]) not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"C4 samples={len(samples)}, completed={len(completed)}, "
        f"pending={len(pending)}"
    )
    if not pending:
        return 0

    generator = Generator(model_path, args)
    model_hash = file_hash(model_path / "config.json")
    written = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as handle:
        for batch_number, batch in enumerate(
            batches(pending, args.batch_size), start=1
        ):
            responses = generator.generate([conversation(item) for item in batch])
            states: list[dict[str, Any]] = []
            retry_indices: list[int] = []
            for index, (sample, response) in enumerate(zip(batch, responses)):
                try:
                    parsed = parse_response(response, sample)
                    error = None
                except ValueError as exc:
                    parsed = None
                    error = str(exc)
                    if args.max_retries:
                        retry_indices.append(index)
                states.append(
                    {
                        "parsed": parsed,
                        "response": response,
                        "error": error,
                        "retry_count": 0,
                    }
                )

            for retry_number in range(1, args.max_retries + 1):
                if not retry_indices:
                    break
                retry_responses = generator.generate(
                    [conversation(batch[index], retry=True) for index in retry_indices]
                )
                remaining: list[int] = []
                for index, response in zip(retry_indices, retry_responses):
                    states[index]["response"] = response
                    states[index]["retry_count"] = retry_number
                    try:
                        states[index]["parsed"] = parse_response(
                            response, batch[index]
                        )
                        states[index]["error"] = None
                    except ValueError as exc:
                        states[index]["error"] = str(exc)
                        remaining.append(index)
                retry_indices = remaining

            for sample, state in zip(batch, states):
                item = record(
                    sample,
                    state["parsed"],
                    state["response"],
                    state["error"],
                    state["retry_count"],
                    args,
                    model_hash,
                )
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
                written += 1
            handle.flush()
            print(
                f"batch={batch_number}, written={written}/{len(pending)}, "
                f"output={output_path}"
            )
    print(f"C4-Explicit complete: {written} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
