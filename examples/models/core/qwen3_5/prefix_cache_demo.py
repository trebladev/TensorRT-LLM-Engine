# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import AutoTokenizer

from tensorrt_llm.runtime import ModelRunnerCpp

_RECURRENT_STATE_SNAPSHOT_INTERVAL = 256
_SUMMARY_SUFFIXES = (
    "总结一下上述文字",
    "总结一下上述内容",
    "总结上述文字",
    "总结上述内容",
)


@dataclass(frozen=True)
class CacheObservation:
    request_reused_blocks: int
    request_hit_rate: float
    cumulative_reused_blocks: int
    cumulative_hit_rate: float
    tokens_per_block: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two sequential requests to validate Qwen3.5 prefix-cache reuse."
    )
    parser.add_argument("--engine_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--input_text", nargs="+", required=True)
    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--max_output_len", type=int, default=512)
    parser.add_argument(
        "--kv_cache_enable_block_reuse",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--kv_cache_free_gpu_memory_fraction", type=float, default=0.2)
    parser.add_argument("--summary_instruction", default="总结上述内容")
    parser.add_argument("--first_max_output_len", type=int, default=1)
    parser.add_argument(
        "--compare_direct_ttft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Create a fresh executor and compare the warm prefix-cache TTFT "
            "against a cold request containing the prefix and summary instruction."
        ),
    )
    args, _ = parser.parse_known_args()
    return args


def remove_trailing_summary_instruction(text: str) -> tuple[str, bool]:
    stripped = text.rstrip()
    for suffix in _SUMMARY_SUFFIXES:
        for punctuation in ("", "。", "！", "!", "？", "?"):
            ending = suffix + punctuation
            if stripped.endswith(ending):
                return stripped[: -len(ending)].rstrip(), True
    return stripped, False


def prepare_requests(
    tokenizer: Any,
    long_text: str,
    summary_instruction: str,
    max_input_length: int,
) -> tuple[list[int], list[int], int, int]:
    long_input_ids = tokenizer.encode(long_text, add_special_tokens=False)
    summary_input_ids = tokenizer.encode("\n\n" + summary_instruction, add_special_tokens=False)
    max_prefix_length = max_input_length - len(summary_input_ids)
    if max_prefix_length <= _RECURRENT_STATE_SNAPSHOT_INTERVAL:
        raise ValueError(
            "max_input_length must leave room for at least 257 prefix tokens "
            "and the summary instruction."
        )

    candidate_length = min(len(long_input_ids), max_prefix_length)
    reusable_boundary = (
        (candidate_length - 1) // _RECURRENT_STATE_SNAPSHOT_INTERVAL
    ) * _RECURRENT_STATE_SNAPSHOT_INTERVAL
    if reusable_boundary < _RECURRENT_STATE_SNAPSHOT_INTERVAL:
        raise ValueError(
            "The long input must provide at least 257 usable prefix tokens to "
            "reach a recurrent-state snapshot boundary."
        )

    # The cache manager stores prompt_length - 1 context tokens. Making the
    # first request 256 * N + 1 tokens commits a complete recurrent snapshot.
    prefix_input_ids = long_input_ids[: reusable_boundary + 1]
    summary_request_ids = prefix_input_ids + summary_input_ids
    return (
        prefix_input_ids,
        summary_request_ids,
        reusable_boundary,
        len(summary_input_ids),
    )


def collect_cache_observation(runner: ModelRunnerCpp) -> CacheObservation:
    request_reused_blocks = 0
    request_hit_rate = 0.0
    for per_iteration in runner.session.get_latest_request_stats():
        for request_stats in per_iteration.request_stats:
            request_reused_blocks = max(
                request_reused_blocks,
                int(request_stats.reused_blocks_per_request),
            )
            request_hit_rate = max(
                request_hit_rate,
                float(request_stats.kv_cache_hit_rate_per_request),
            )

    cumulative_reused_blocks = 0
    cumulative_hit_rate = 0.0
    tokens_per_block = 0
    for iteration_stats in runner.session.get_latest_iteration_stats():
        kv_cache_stats = iteration_stats.kv_cache_stats
        cumulative_reused_blocks = int(kv_cache_stats.reused_blocks)
        cumulative_hit_rate = float(kv_cache_stats.cache_hit_rate)
        tokens_per_block = int(kv_cache_stats.tokens_per_block)

    return CacheObservation(
        request_reused_blocks=request_reused_blocks,
        request_hit_rate=request_hit_rate,
        cumulative_reused_blocks=cumulative_reused_blocks,
        cumulative_hit_rate=cumulative_hit_rate,
        tokens_per_block=tokens_per_block,
    )


def create_runner(args: argparse.Namespace) -> ModelRunnerCpp:
    return ModelRunnerCpp.from_dir(
        engine_dir=args.engine_dir,
        max_batch_size=1,
        max_input_len=args.max_input_length,
        max_output_len=args.max_output_len,
        max_beam_width=1,
        kv_cache_enable_block_reuse=True,
        kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
    )


def run_streaming_request(
    runner: ModelRunnerCpp,
    input_ids: list[int],
    max_new_tokens: int,
    end_id: int,
    pad_id: int,
) -> tuple[dict[str, torch.Tensor], float, float]:
    request = torch.tensor(input_ids, dtype=torch.int32)
    start_time = perf_counter()
    first_token_latency = None
    final_outputs = None
    with torch.no_grad():
        output_stream = runner.generate(
            batch_input_ids=[request],
            max_new_tokens=max_new_tokens,
            end_id=end_id,
            pad_id=pad_id,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_beams=1,
            return_dict=True,
            output_sequence_lengths=True,
            streaming=True,
        )
        for outputs in output_stream:
            torch.cuda.synchronize()
            sequence_length = int(outputs["sequence_lengths"][0][0].item())
            if first_token_latency is None and sequence_length > len(input_ids):
                first_token_latency = perf_counter() - start_time
            final_outputs = outputs

    if first_token_latency is None or final_outputs is None:
        raise RuntimeError("The streaming request completed without producing a token.")
    return final_outputs, first_token_latency, perf_counter() - start_time


def warmup_runner(
    runner: ModelRunnerCpp,
    warmup_ids: list[int],
    end_id: int,
    pad_id: int,
) -> None:
    run_streaming_request(
        runner,
        warmup_ids,
        max_new_tokens=1,
        end_id=end_id,
        pad_id=pad_id,
    )
    collect_cache_observation(runner)


def decode_generated_text(
    tokenizer: Any,
    outputs: dict[str, torch.Tensor],
    input_length: int,
) -> str:
    sequence_length = int(outputs["sequence_lengths"][0][0].item())
    generated_ids = outputs["output_ids"][0][0][input_length:sequence_length].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_arguments()
    if not args.kv_cache_enable_block_reuse:
        raise ValueError("Prefix-cache validation requires --kv_cache_enable_block_reuse.")
    if args.first_max_output_len < 1:
        raise ValueError("first_max_output_len must be positive.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_text = " ".join(args.input_text)
    long_text, removed_summary = remove_trailing_summary_instruction(input_text)
    (
        first_request_ids,
        second_request_ids,
        reusable_boundary,
        summary_suffix_length,
    ) = prepare_requests(
        tokenizer,
        long_text,
        args.summary_instruction,
        args.max_input_length,
    )

    print("=== Qwen3.5 prefix-cache validation ===")
    print(f"Removed summary instruction from original input: {removed_summary}")
    print(f"Request 1 prefix tokens: {len(first_request_ids)}")
    print(f"Request 2 summary suffix tokens: {summary_suffix_length}")
    print(f"Expected reusable token boundary: {reusable_boundary}")

    warmup_ids = second_request_ids.copy()
    warmup_ids[0] = (warmup_ids[0] + 1) % len(tokenizer)

    runner = create_runner(args)
    warmup_runner(
        runner,
        warmup_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )

    first_outputs, first_ttft, first_latency = run_streaming_request(
        runner,
        first_request_ids,
        args.first_max_output_len,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )
    first_observation = collect_cache_observation(runner)
    first_output_text = decode_generated_text(tokenizer, first_outputs, len(first_request_ids))
    print("\n--- Request 1: populate cache ---")
    print(f"TTFT: {first_ttft * 1000:.2f} ms")
    print(f"Total latency: {first_latency:.3f} s")
    print(f"Generated text: {first_output_text!r}")
    print(f"Request reused blocks: {first_observation.request_reused_blocks}")

    second_outputs, second_ttft, second_latency = run_streaming_request(
        runner,
        second_request_ids,
        args.max_output_len,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )
    second_observation = collect_cache_observation(runner)
    summary_text = decode_generated_text(tokenizer, second_outputs, len(second_request_ids))
    cumulative_reuse_delta = max(
        0,
        second_observation.cumulative_reused_blocks - first_observation.cumulative_reused_blocks,
    )
    observed_reused_blocks = max(second_observation.request_reused_blocks, cumulative_reuse_delta)

    print("\n--- Request 2: reuse prefix and summarize ---")
    print(f"TTFT: {second_ttft * 1000:.2f} ms")
    print(f"Total latency: {second_latency:.3f} s")
    print(f"Summary:\n{summary_text}")
    print(f"Request-level reused blocks (if retained): {second_observation.request_reused_blocks}")
    print(f"Request-level cache hit rate (if retained): {second_observation.request_hit_rate:.2%}")
    print(f"Cumulative reused-block delta: {cumulative_reuse_delta}")
    print(f"Cumulative cache hit rate: {second_observation.cumulative_hit_rate:.2%}")
    print(f"Observed reused blocks: {observed_reused_blocks}")
    if second_observation.tokens_per_block > 0:
        expected_attention_blocks = reusable_boundary // second_observation.tokens_per_block
        print(f"KV tokens per block: {second_observation.tokens_per_block}")
        print(f"Expected reusable attention blocks: {expected_attention_blocks}")

    if observed_reused_blocks == 0:
        print(
            "WARNING: no reused cache blocks were observed. Check that the "
            "engine uses paged KV cache and that block reuse is enabled."
        )
    else:
        print(f"PASS: observed {observed_reused_blocks} reused cache blocks.")

    if args.compare_direct_ttft:
        runner.session.shutdown()
        del runner
        torch.cuda.empty_cache()

        direct_runner = create_runner(args)
        warmup_runner(
            direct_runner,
            warmup_ids,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        direct_outputs, direct_ttft, direct_latency = run_streaming_request(
            direct_runner,
            second_request_ids,
            args.max_output_len,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        direct_observation = collect_cache_observation(direct_runner)
        direct_runner.session.shutdown()
        direct_summary = decode_generated_text(tokenizer, direct_outputs, len(second_request_ids))
        ttft_reduction = direct_ttft - second_ttft
        ttft_reduction_percent = ttft_reduction / direct_ttft * 100 if direct_ttft > 0 else 0.0
        ttft_speedup = direct_ttft / second_ttft if second_ttft > 0 else 0.0

        print("\n--- Direct combined input: fresh executor baseline ---")
        print(f"TTFT: {direct_ttft * 1000:.2f} ms")
        print(f"Total latency: {direct_latency:.3f} s")
        print(f"Generated text:\n{direct_summary}")
        print(f"Cold baseline reused blocks: {direct_observation.cumulative_reused_blocks}")
        print("\n--- TTFT comparison ---")
        print("Each executor used one untimed, nonmatching long warmup request.")
        print(f"Request 1 cold prefix TTFT: {first_ttft * 1000:.2f} ms")
        print(f"Request 2 warm prefix TTFT: {second_ttft * 1000:.2f} ms")
        print(f"Direct combined cold TTFT: {direct_ttft * 1000:.2f} ms")
        print(f"Warm-prefix TTFT reduction: {ttft_reduction * 1000:.2f} ms")
        print(f"Warm-prefix TTFT reduction: {ttft_reduction_percent:.2f}%")
        print(f"Warm-prefix TTFT speedup: {ttft_speedup:.2f}x")


if __name__ == "__main__":
    main()
