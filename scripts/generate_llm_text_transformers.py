"""Generate MOSI/MOSEI enhancements without vLLM or FlashInfer.

This entry point uses Hugging Face Transformers with PyTorch SDPA. It reuses
the prompt, context construction, JSON parsing, fallback, and checkpointing
helpers from generate_llm_text.py, so both backends can safely target the same
append-only JSONL file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from generate_llm_text import (
    batched,
    build_conversation,
    build_samples,
    file_sha256,
    load_completed_records,
    load_dataset,
    make_record,
    parse_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5 rewrites with Transformers + PyTorch SDPA."
    )
    parser.add_argument(
        "--data-path",
        default="datasets/mosi/unaligned_50.pkl",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Qwen2.5-Instruct directory on AutoDL.",
    )
    parser.add_argument(
        "--output-path",
        default="datasets/mosi/qwen25_enhanced.jsonl",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid", "test"],
    )
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Reduce to 4 or 2 if CUDA runs out of memory.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace the JSONL instead of resuming it.",
    )
    return parser.parse_args()


class TransformersGenerator:
    def __init__(self, model_path: Path, args: argparse.Namespace):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "This backend requires torch, transformers, and accelerate."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for Qwen2.5 generation")

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
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
            "dtype": dtype_map[args.dtype],
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path), **model_kwargs
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            # Transformers 4.x used torch_dtype; 5.x uses dtype.
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path), **model_kwargs
            )

        self.model.eval()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(
            "Loaded Transformers backend with PyTorch SDPA; "
            "vLLM/FlashInfer/CUTLASS are not used."
        )

    def generate(self, conversations: list[list[dict[str, str]]]) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        model_inputs = {
            key: value.to(self.model.device) for key, value in model_inputs.items()
        }
        input_length = model_inputs["input_ids"].shape[1]

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=self.max_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[:, input_length:]
        return self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_input_tokens <= args.max_tokens:
        raise ValueError("--max-input-tokens must be greater than --max-tokens")

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")

    data = load_dataset(data_path)
    samples = build_samples(data, args.splits, args.context_window)

    if args.overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed, malformed_lines = load_completed_records(output_path)
    if malformed_lines:
        print(
            f"Warning: ignored {malformed_lines} malformed existing JSONL lines.",
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

    generator = TransformersGenerator(model_path, args)
    model_config_hash = file_sha256(model_path / "config.json")

    written = 0
    with output_path.open("a", encoding="utf-8", buffering=1) as output_handle:
        for batch_number, batch in enumerate(
            batched(pending, args.batch_size), start=1
        ):
            conversations = [build_conversation(sample) for sample in batch]
            responses = generator.generate(conversations)
            if len(responses) != len(batch):
                raise RuntimeError(
                    "Transformers returned a different number of responses"
                )

            states: list[dict[str, Any]] = []
            retry_indices: list[int] = []
            for index, response in enumerate(responses):
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
                retry_responses = generator.generate(retry_conversations)
                next_retry_indices: list[int] = []
                for original_index, response in zip(
                    retry_indices, retry_responses
                ):
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
                record["generation"]["backend"] = "transformers"
                record["generation"]["attention"] = "sdpa"
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
