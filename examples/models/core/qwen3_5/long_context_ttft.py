# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import torch
from transformers import AutoTokenizer

from tensorrt_llm.runtime import ModelRunnerCpp

_RECURRENT_STATE_SNAPSHOT_INTERVAL = 256


@dataclass(frozen=True)
class CacheObservation:
    cumulative_reused_blocks: int
    cumulative_missed_blocks: int
    tokens_per_block: int


@dataclass(frozen=True)
class RequestMeasurement:
    ttft_seconds: float
    total_latency_seconds: float
    reused_blocks: int
    missed_blocks: int
    cache_hit_rate: float


@dataclass(frozen=True)
class TrialMeasurement:
    order: str
    prefix_population: RequestMeasurement
    cold_long: RequestMeasurement
    prefix_hit_long: RequestMeasurement
    tokens_per_block: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3.5 TTFT at 50K and 100K input lengths."
    )
    parser.add_argument("--engine_dir", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--prefix_tokens", type=int, default=50 * 1024)
    parser.add_argument("--long_tokens", type=int, default=100 * 1024)
    parser.add_argument("--max_output_len", type=int, default=8)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--warmup_requests", type=int, default=1)
    parser.add_argument(
        "--request_order",
        choices=("alternating", "cold_first", "hit_first"),
        default="alternating",
        help="Order of the measured cold and prefix-hit requests within each executor.",
    )
    parser.add_argument(
        "--kv_cache_enable_block_reuse",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--kv_cache_free_gpu_memory_fraction", type=float, default=0.7)
    args, _ = parser.parse_known_args()
    return args


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.kv_cache_enable_block_reuse:
        raise ValueError("The benchmark requires --kv_cache_enable_block_reuse.")
    if args.prefix_tokens <= 0 or args.long_tokens <= 0:
        raise ValueError("prefix_tokens and long_tokens must be positive.")
    if args.prefix_tokens % _RECURRENT_STATE_SNAPSHOT_INTERVAL != 0:
        raise ValueError(
            "prefix_tokens must be divisible by the 256-token recurrent-state snapshot interval."
        )
    if args.long_tokens <= args.prefix_tokens + 1:
        raise ValueError("long_tokens must be greater than prefix_tokens + 1.")
    if args.max_output_len < 2:
        raise ValueError(
            "max_output_len must be at least 2 so TTFT uses a non-final streaming response."
        )
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if args.warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative.")


def make_input_ids(tokenizer: AutoTokenizer, token_count: int) -> list[int]:
    seed_ids = tokenizer.encode(
        "TensorRT-LLM Qwen3.5 long-context prefix-cache TTFT benchmark. ",
        add_special_tokens=False,
    )
    if not seed_ids:
        raise ValueError("The tokenizer produced no seed tokens.")
    repeat_count = (token_count + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeat_count)[:token_count]


def make_nonmatching_ids(
    input_ids: list[int], vocab_size: int, first_token_offset: int
) -> list[int]:
    if not input_ids:
        raise ValueError("The nonmatching input must not be empty.")
    if vocab_size <= 1:
        raise ValueError("The tokenizer vocabulary must contain at least two tokens.")
    nonmatching_ids = input_ids.copy()
    offset = first_token_offset % vocab_size
    if offset == 0:
        offset = 1
    nonmatching_ids[0] = (nonmatching_ids[0] + offset) % vocab_size
    return nonmatching_ids


def create_runner(args: argparse.Namespace) -> ModelRunnerCpp:
    return ModelRunnerCpp.from_dir(
        engine_dir=args.engine_dir,
        max_batch_size=1,
        max_input_len=args.long_tokens,
        max_output_len=args.max_output_len,
        max_beam_width=1,
        kv_cache_enable_block_reuse=True,
        kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
        enable_chunked_context=True,
        is_orchestrator_mode=True,
        device_ids=[0, 1],
    )


def collect_cache_observation(runner: ModelRunnerCpp) -> CacheObservation:
    runner.session.get_latest_request_stats()
    cumulative_reused_blocks = 0
    cumulative_missed_blocks = 0
    tokens_per_block = 0
    for iteration_stats in runner.session.get_latest_iteration_stats():
        kv_cache_stats = iteration_stats.kv_cache_stats
        cumulative_reused_blocks = int(kv_cache_stats.reused_blocks)
        cumulative_missed_blocks = int(kv_cache_stats.missed_blocks)
        tokens_per_block = int(kv_cache_stats.tokens_per_block)
    return CacheObservation(
        cumulative_reused_blocks=cumulative_reused_blocks,
        cumulative_missed_blocks=cumulative_missed_blocks,
        tokens_per_block=tokens_per_block,
    )


def run_streaming_request(
    runner: ModelRunnerCpp,
    input_ids: list[int],
    max_new_tokens: int,
    end_id: int,
    pad_id: int,
) -> tuple[float, float]:
    request = torch.tensor(input_ids, dtype=torch.int32)
    first_token_latency = None
    received_output = False
    start_time = perf_counter()
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
            received_output = True

    if first_token_latency is None or not received_output:
        raise RuntimeError("The streaming request completed without producing a token.")
    return first_token_latency, perf_counter() - start_time


def warmup_runner(
    runner: ModelRunnerCpp,
    long_input_ids: list[int],
    args: argparse.Namespace,
    vocab_size: int,
    end_id: int,
    pad_id: int,
) -> CacheObservation:
    for warmup_index in range(args.warmup_requests):
        warmup_ids = make_nonmatching_ids(
            long_input_ids,
            vocab_size,
            first_token_offset=warmup_index + 2,
        )
        run_streaming_request(
            runner,
            warmup_ids,
            args.max_output_len,
            end_id,
            pad_id,
        )
    return collect_cache_observation(runner)


def measure_request(
    runner: ModelRunnerCpp,
    input_ids: list[int],
    args: argparse.Namespace,
    end_id: int,
    pad_id: int,
    previous_observation: CacheObservation,
) -> tuple[RequestMeasurement, CacheObservation]:
    ttft_seconds, total_latency_seconds = run_streaming_request(
        runner,
        input_ids,
        args.max_output_len,
        end_id,
        pad_id,
    )
    observation = collect_cache_observation(runner)
    reused_blocks = max(
        0,
        observation.cumulative_reused_blocks - previous_observation.cumulative_reused_blocks,
    )
    missed_blocks = max(
        0,
        observation.cumulative_missed_blocks - previous_observation.cumulative_missed_blocks,
    )
    total_blocks = reused_blocks + missed_blocks
    cache_hit_rate = reused_blocks / total_blocks if total_blocks > 0 else 0.0
    return (
        RequestMeasurement(
            ttft_seconds=ttft_seconds,
            total_latency_seconds=total_latency_seconds,
            reused_blocks=reused_blocks,
            missed_blocks=missed_blocks,
            cache_hit_rate=cache_hit_rate,
        ),
        observation,
    )


def resolve_trial_order(request_order: str, trial_index: int) -> str:
    if request_order == "alternating":
        return "cold_first" if trial_index % 2 == 0 else "hit_first"
    return request_order


def run_trial(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    long_input_ids: list[int],
    cold_input_ids: list[int],
    prefix_population_ids: list[int],
    trial_index: int,
) -> TrialMeasurement:
    order = resolve_trial_order(args.request_order, trial_index)
    runner = create_runner(args)
    try:
        observation = warmup_runner(
            runner,
            long_input_ids,
            args,
            len(tokenizer),
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )

        if order == "cold_first":
            cold_long, observation = measure_request(
                runner,
                cold_input_ids,
                args,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                observation,
            )
            prefix_population, observation = measure_request(
                runner,
                prefix_population_ids,
                args,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                observation,
            )
            prefix_hit_long, observation = measure_request(
                runner,
                long_input_ids,
                args,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                observation,
            )
        else:
            prefix_population, observation = measure_request(
                runner,
                prefix_population_ids,
                args,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                observation,
            )
            prefix_hit_long, observation = measure_request(
                runner,
                long_input_ids,
                args,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                observation,
            )
            cold_long, observation = measure_request(
                runner,
                cold_input_ids,
                args,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                observation,
            )

        return TrialMeasurement(
            order=order,
            prefix_population=prefix_population,
            cold_long=cold_long,
            prefix_hit_long=prefix_hit_long,
            tokens_per_block=observation.tokens_per_block,
        )
    finally:
        runner.session.shutdown()
        del runner
        torch.cuda.empty_cache()


def print_measurement(label: str, token_count: int, result: RequestMeasurement) -> None:
    print(f"\n--- {label} ---")
    print(f"Input tokens: {token_count}")
    print(f"TTFT: {result.ttft_seconds * 1000:.2f} ms")
    print(f"Total latency: {result.total_latency_seconds:.3f} s")
    print(f"Reused blocks: {result.reused_blocks}")
    print(f"Missed blocks: {result.missed_blocks}")
    print(f"Request cache hit rate: {result.cache_hit_rate:.2%}")


def print_trial(
    trial_index: int,
    prefix_tokens: int,
    long_tokens: int,
    trial: TrialMeasurement,
) -> None:
    order_label = (
        "cold -> prefix -> hit" if trial.order == "cold_first" else "prefix -> hit -> cold"
    )
    print(f"\n=== Trial {trial_index + 1}: {order_label} ===")
    print_measurement(
        "50K cold prefix population",
        prefix_tokens + 1,
        trial.prefix_population,
    )
    print_measurement("100K cold input", long_tokens, trial.cold_long)
    print_measurement(
        "100K input with 50K prefix hit",
        long_tokens,
        trial.prefix_hit_long,
    )
    speedup = trial.cold_long.ttft_seconds / trial.prefix_hit_long.ttft_seconds
    print(f"Trial TTFT speedup: {speedup:.2f}x")


def format_distribution(values: list[float], scale: float = 1.0) -> str:
    scaled_values = [value * scale for value in values]
    return (
        f"median={median(scaled_values):.2f}, "
        f"min={min(scaled_values):.2f}, max={max(scaled_values):.2f}"
    )


def validate_trial(trial: TrialMeasurement) -> None:
    if trial.prefix_population.reused_blocks != 0:
        raise RuntimeError("A prefix population request unexpectedly reused cache blocks.")
    if trial.cold_long.reused_blocks != 0:
        raise RuntimeError("A cold 100K request unexpectedly reused cache blocks.")
    if trial.prefix_hit_long.reused_blocks == 0:
        raise RuntimeError("A 100K request did not reuse its 50K prefix.")


def print_summary(args: argparse.Namespace, trials: list[TrialMeasurement]) -> None:
    prefix_ttfts = [trial.prefix_population.ttft_seconds for trial in trials]
    cold_ttfts = [trial.cold_long.ttft_seconds for trial in trials]
    hit_ttfts = [trial.prefix_hit_long.ttft_seconds for trial in trials]
    trial_speedups = [
        trial.cold_long.ttft_seconds / trial.prefix_hit_long.ttft_seconds for trial in trials
    ]
    reused_blocks = [float(trial.prefix_hit_long.reused_blocks) for trial in trials]

    median_cold = median(cold_ttfts)
    median_hit = median(hit_ttfts)
    reduction = median_cold - median_hit
    reduction_percent = reduction / median_cold * 100 if median_cold > 0 else 0.0
    speedup = median_cold / median_hit if median_hit > 0 else 0.0

    print("\n=== Aggregate TTFT summary ===")
    print(f"Trials: {len(trials)}")
    print(f"Warmup requests per trial: {args.warmup_requests}")
    print(f"50K cold TTFT (ms): {format_distribution(prefix_ttfts, scale=1000)}")
    print(f"100K cold TTFT (ms): {format_distribution(cold_ttfts, scale=1000)}")
    print(f"100K with 50K hit TTFT (ms): {format_distribution(hit_ttfts, scale=1000)}")
    print(f"Per-trial speedup: {format_distribution(trial_speedups)}")
    print(f"Median TTFT reduction: {reduction * 1000:.2f} ms")
    print(f"Median TTFT reduction: {reduction_percent:.2f}%")
    print(f"Median TTFT speedup: {speedup:.2f}x")
    print(f"Hit reused blocks: {format_distribution(reused_blocks)}")

    tokens_per_block = trials[-1].tokens_per_block
    if tokens_per_block > 0:
        attention_blocks = args.prefix_tokens // tokens_per_block
        recurrent_snapshots = args.prefix_tokens // _RECURRENT_STATE_SNAPSHOT_INTERVAL
        print(f"Expected full-attention blocks: {attention_blocks}")
        print(f"Expected recurrent-state snapshots: {recurrent_snapshots}")
        print(f"Expected total reused blocks: {attention_blocks + recurrent_snapshots}")
    print("PASS: every trial kept cold requests cold and reused the complete 50K prefix.")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    long_input_ids = make_input_ids(tokenizer, args.long_tokens)
    cold_input_ids = make_nonmatching_ids(
        long_input_ids,
        len(tokenizer),
        first_token_offset=1,
    )
    prefix_population_ids = long_input_ids[: args.prefix_tokens + 1]

    print("=== Qwen3.5 unified long-context TTFT benchmark ===")
    print(f"50K reusable boundary: {args.prefix_tokens} tokens")
    print(f"50K population request: {len(prefix_population_ids)} tokens")
    print(f"100K request: {len(long_input_ids)} tokens")
    print(f"Trials: {args.num_trials}")
    print(f"Warmup requests per trial: {args.warmup_requests}")
    print(f"Request order: {args.request_order}")

    trials = []
    for trial_index in range(args.num_trials):
        trial = run_trial(
            args,
            tokenizer,
            long_input_ids,
            cold_input_ids,
            prefix_population_ids,
            trial_index,
        )
        validate_trial(trial)
        trials.append(trial)
        print_trial(
            trial_index,
            args.prefix_tokens,
            args.long_tokens,
            trial,
        )

    print_summary(args, trials)


if __name__ == "__main__":
    main()
