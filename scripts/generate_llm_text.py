"""Generate conservative Qwen2.5 text enhancements for MOSI/MOSEI.

The script never reads sentiment labels. It groups utterances by video id,
uses neighbouring utterances from the same split as context, and writes one
auditable JSON object per sample. Output is append-only by default so an
interrupted AutoDL run can be resumed safely.
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


PROMPT_VERSION = "qwen25-semantic-rewrite-v1"

SYSTEM_PROMPT = """You are a conservative semantic rewriter for spoken English transcripts.

Rewrite only the TARGET utterance by:
1. resolving pronouns and references when the supplied context supports it;
2. disambiguating expressions only when the context supports one reliable interpretation;
3. making implicit semantic content explicit;
4. removing unnecessary speech disfluencies when appropriate.

Strict requirements:
- Preserve the original sentiment polarity and sentiment intensity.
- Do not introduce new facts, entities, opinions, causes, or sentiment.
- Use only the supplied transcript. Never output a sentiment label or score.
- If a reference cannot be resolved reliably, retain the original expression.
- Keep the result concise, preferably no more than 40 English words.
- Rewrite TARGET only; do not rewrite or quote the context.
- Return exactly one valid JSON object and no Markdown."""

USER_TEMPLATE = """Rewrite the TARGET utterance.

PREVIOUS CONTEXT:
{previous_context}

TARGET:
{target}

FOLLOWING CONTEXT:
{following_context}

Return this JSON schema:
{{
  "enhanced_text": "rewritten TARGET",
  "coreference_completed": true,
  "ambiguity_resolved": false,
  "implicit_meaning_explicit": true,
  "uncertain": false
}}"""

RETRY_SUFFIX = """

Your previous response could not be parsed. Return one JSON object only.
Do not add a code fence, explanation, preface, or trailing text."""

BOOL_FIELDS = (
    "coreference_completed",
    "ambiguity_resolved",
    "implicit_meaning_explicit",
    "uncertain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label-free Qwen2.5 semantic rewrites for MOSI/MOSEI."
    )
    parser.add_argument(
        "--data-path",
        default="datasets/mosi/unaligned_50.pkl",
        help="Original MMSA-format PKL. It is opened read-only.",
    )
    parser.add_argument(
        "--model-path",
        default="../Qwen2.5-7B-Instruct",
        help="Local Qwen2.5-Instruct directory on AutoDL.",
    )
    parser.add_argument(
        "--output-path",
        default="datasets/mosi/qwen25_enhanced.jsonl",
        help="Append-only enhancement records.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid", "test"],
        help="Tune/freeze the prompt on train/valid before generating test.",
    )
    parser.add_argument("--context-window", type=int, default=2)
    #设置拿到上下两个上下文窗口的文本
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries for malformed JSON; inference errors are not hidden.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate at most N pending samples (useful for a pilot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt examples without loading Qwen or writing output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing JSONL instead of resuming it.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", str(value)).strip()


def split_segment_id(sample_id: str) -> tuple[str, int | None]:
    if "$_$" not in sample_id:
        return sample_id, None
    video_id, segment = sample_id.rsplit("$_$", 1)
    try:
        return video_id, int(segment)
    except ValueError:
        return video_id, None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    for split in ("train", "valid", "test"):
        if split not in data:
            raise KeyError(f"Dataset has no '{split}' split")
        for key in ("id", "raw_text"):
            if key not in data[split]:
                raise KeyError(f"Dataset split '{split}' has no '{key}' field")
    return data


def build_samples(
    data: dict[str, Any], splits: Iterable[str], context_window: int
) -> list[dict[str, Any]]:
    if context_window < 0:
        raise ValueError("--context-window must be >= 0")

    samples: list[dict[str, Any]] = []
    for split in splits:
        ids = [normalize_text(value) for value in data[split]["id"]]
        texts = [normalize_text(value) for value in data[split]["raw_text"]]
        if len(ids) != len(texts):
            raise ValueError(f"{split}: id/raw_text length mismatch")

        groups: dict[str, list[tuple[int | None, int]]] = defaultdict(list)
        for index, sample_id in enumerate(ids):
            video_id, segment_number = split_segment_id(sample_id)
            groups[video_id].append((segment_number, index))

        contexts: dict[int, tuple[list[int], list[int]]] = {}
        for members in groups.values():
            members.sort(
                key=lambda item: (
                    item[0] is None,
                    item[0] if item[0] is not None else item[1],
                    item[1],
                )
            )
            ordered_indices = [item[1] for item in members]
            for position, index in enumerate(ordered_indices):
                before = ordered_indices[max(0, position - context_window) : position]
                after = ordered_indices[
                    position + 1 : position + 1 + context_window
                ]
                contexts[index] = (before, after)

        for index, (sample_id, raw_text) in enumerate(zip(ids, texts)):
            before_indices, after_indices = contexts[index]
            samples.append(
                {
                    "split": split,
                    "index": index,
                    "id": sample_id,
                    "raw_text": raw_text,
                    "context_before_ids": [ids[i] for i in before_indices],
                    "context_before": [texts[i] for i in before_indices],
                    "context_after_ids": [ids[i] for i in after_indices],
                    "context_after": [texts[i] for i in after_indices],
                }
            )
    return samples


def format_context(texts: list[str]) -> str:
    if not texts:
        return "<NONE>"
    return "\n".join(f"- {text}" for text in texts)


def build_conversation(sample: dict[str, Any], retry: bool = False) -> list[dict[str, str]]:
    user_prompt = USER_TEMPLATE.format(
        previous_context=format_context(sample["context_before"]),
        target=sample["raw_text"],
        following_context=format_context(sample["context_after"]),
    )
    if retry:
        user_prompt += RETRY_SUFFIX
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(response: str) -> dict[str, Any]:
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

    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    if start >= 0:
        try:
            value, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("response does not contain a valid JSON object")


def parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"'{field}' must be a JSON boolean")


def parse_response(response: str) -> dict[str, Any]:
    parsed = extract_json_object(response)
    enhanced_text = normalize_text(parsed.get("enhanced_text", ""))
    if not enhanced_text:
        raise ValueError("'enhanced_text' is empty")
    if len(enhanced_text) > 1000:
        raise ValueError("'enhanced_text' is unexpectedly long")

    result: dict[str, Any] = {"enhanced_text": enhanced_text}
    for field in BOOL_FIELDS:
        result[field] = parse_bool(parsed.get(field), field)
    return result


def load_completed_records(output_path: Path) -> tuple[set[tuple[str, str]], int]:
    completed: set[tuple[str, str]] = set()
    malformed = 0
    if not output_path.is_file():
        return completed, malformed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (str(record["split"]), normalize_text(record["id"]))
                completed.add(key)
            except (json.JSONDecodeError, KeyError, TypeError):
                malformed += 1
    return completed, malformed


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("--batch-size must be > 0")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def make_record(
    sample: dict[str, Any],
    parsed: dict[str, Any] | None,
    raw_response: str,
    model_path: str,
    model_config_hash: str | None,
    retry_count: int,
    error: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if parsed is None:
        parsed = {
            "enhanced_text": sample["raw_text"],
            "coreference_completed": False,
            "ambiguity_resolved": False,
            "implicit_meaning_explicit": False,
            "uncertain": True,
        }
        status = "fallback_parse_error"
    else:
        status = "success_after_retry" if retry_count else "success"

    return {
        "split": sample["split"],
        "index": sample["index"],
        "id": sample["id"],
        "raw_text": sample["raw_text"],
        "enhanced_text": parsed["enhanced_text"],
        "coreference_completed": parsed["coreference_completed"],
        "ambiguity_resolved": parsed["ambiguity_resolved"],
        "implicit_meaning_explicit": parsed["implicit_meaning_explicit"],
        "uncertain": parsed["uncertain"],
        "context_before_ids": sample["context_before_ids"],
        "context_after_ids": sample["context_after_ids"],
        "status": status,
        "error": error,
        "retry_count": retry_count,
        "raw_response": raw_response,
        "source_text_sha256": sha256_text(sample["raw_text"]),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT + "\n" + USER_TEMPLATE),
        "model_path": model_path,
        "model_config_sha256": model_config_hash,
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    data = load_dataset(data_path)
    samples = build_samples(data, args.splits, args.context_window)

    if args.dry_run:
        preview_count = min(args.limit or 3, len(samples))
        print(f"Dry run: {len(samples)} samples available; showing {preview_count}.")
        for sample in samples[:preview_count]:
            print("=" * 80)
            print(f"{sample['split']} / {sample['id']}")
            print(json.dumps(build_conversation(sample), ensure_ascii=False, indent=2))
        return 0

    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}. "
            "Pass the actual AutoDL Qwen2.5-Instruct path via --model-path."
        )

    if args.overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed, malformed_lines = load_completed_records(output_path)
    if malformed_lines:
        print(
            f"Warning: {malformed_lines} malformed existing JSONL lines were ignored.",
            file=sys.stderr,
        )
    pending = [
        sample
        for sample in samples
        if (sample["split"], sample["id"]) not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"Dataset samples={len(samples)}, completed={len(completed)}, "
        f"pending this run={len(pending)}"
    )
    if not pending:
        return 0

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is required on AutoDL. Install a vLLM build compatible with "
            "the server's PyTorch/CUDA image."
        ) from exc

    llm = LLM(
        model=str(model_path),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        seed=args.seed,
        skip_special_tokens=True,
    )
    model_config_hash = file_sha256(model_path / "config.json")

    written = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as output_handle:
        for batch_number, batch in enumerate(
            batched(pending, args.batch_size), start=1
        ):
            conversations = [build_conversation(sample) for sample in batch]
            outputs = llm.chat(
                conversations,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of outputs")

            states: list[dict[str, Any]] = []
            retry_indices: list[int] = []
            for index, output in enumerate(outputs):
                response = output.outputs[0].text
                try:
                    parsed = parse_response(response)
                    error = None
                except ValueError as exc:
                    parsed = None
                    error = str(exc)
                    if args.max_retries > 0:
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
                retry_conversations = [
                    build_conversation(batch[index], retry=True)
                    for index in retry_indices
                ]
                retry_outputs = llm.chat(
                    retry_conversations,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                next_retry_indices: list[int] = []
                for original_index, output in zip(retry_indices, retry_outputs):
                    response = output.outputs[0].text
                    states[original_index]["response"] = response
                    states[original_index]["retry_count"] = retry_number
                    try:
                        states[original_index]["parsed"] = parse_response(response)
                        states[original_index]["error"] = None
                    except ValueError as exc:
                        states[original_index]["error"] = str(exc)
                        next_retry_indices.append(original_index)
                retry_indices = next_retry_indices

            for sample, state in zip(batch, states):
                record = make_record(
                    sample=sample,
                    parsed=state["parsed"],
                    raw_response=state["response"],
                    model_path=str(model_path),
                    model_config_hash=model_config_hash,
                    retry_count=state["retry_count"],
                    error=state["error"],
                    args=args,
                )
                output_handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                written += 1

            output_handle.flush()
            print(
                f"batch={batch_number}, written={written}/{len(pending)}, "
                f"output={output_path}"
            )

    print(f"Generation complete: {written} new records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
