#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Fusion Linear CE against four unfused PyTorch expressions.

This is an HCU operator benchmark, not an ordinary CPU unit test.  Run it in
an environment with a compatible HCU PyTorch, gfx936 device, and installed
``hcu_linear_ce_artifacts_gfx936`` wheel, for example::

    cd /public/home/yugeng/work/Megatron-LM-das
    export PYTHONPATH="$PWD"
    export HIP_VISIBLE_DEVICES=0
    python3 tests/unit_tests/fusions/benchmark_hcu_linear_cross_entropy.py \
        --n 8192 --d 4096 --v 32000 --warmup 1 --iterations 5

The five measured paths are:

1. ``fusion_bf16``
   The public Megatron ``linear_cross_entropy`` API and its native .so.
2. ``unfused_bf16_linear_logsumexp``
   BF16 ``F.linear`` followed by a correct FP32 ``logsumexp`` CE formula.
   This is the closest valid unfused counterpart to Fusion's BF16 input path.
3. ``unfused_fp32_linear_logsumexp``
   The same formula, but its linear projection is FP32.  It is the numerical
   reference used by the Linear CE validation work.
4. ``pytorch_bf16_linear_cross_entropy``
   The conventional mixed-precision expression: BF16 F.linear followed by
   HCU F.cross_entropy.  This is the standard-PyTorch BF16 counterpart to
   Fusion's BF16 input path.
5. ``pytorch_fp32_linear_cross_entropy``
   The familiar ``F.linear(...); F.cross_entropy(...)`` expression.  It is a
   valid FP32 PyTorch baseline and also quantifies the cost of materializing
   FP32 logits.

All inputs are derived from one fixed-seed BF16 hidden/weight/label set.
Each timed sample reports two complementary forward+backward measurements:

* device time: HCU stream elapsed time from torch.cuda.Event;
* host time: perf_counter wall-clock time through the final event synchronize.

Device time is the primary operator metric.  Host time additionally includes
Python dispatch and kernel-launch overhead, and is useful for framework-side
end-to-end observation.  Peak memory is the incremental PyTorch allocator peak
after the persistent input tensors and one warmup have been established;
allocator-reserved memory is also reported but is cache-sensitive, so allocated
peak is the primary comparison metric.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from hcu_megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy


IGNORE_INDEX = -100
SEED = 20260903


@dataclass
class BenchmarkResult:
    """Measurements from one forward+backward implementation."""

    name: str
    loss: float
    dhidden_finite: bool
    dweight_finite: bool
    device_median_ms: float
    host_median_ms: float
    host_mean_ms: float
    host_min_ms: float
    host_max_ms: float
    extra_peak_allocated_mib: float
    extra_peak_reserved_mib: float
    persistent_input_mib: float
    device_samples_ms: list[float]
    host_samples_ms: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8192, help="Number of tokens.")
    parser.add_argument("--d", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--v", type=int, default=32000, help="Vocabulary size.")
    parser.add_argument("--device", type=int, default=0, help="Logical HCU device index.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed iterations per path.")
    parser.add_argument("--iterations", type=int, default=5, help="Timed iterations per path.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.n, args.d, args.v) <= 0:
        raise ValueError("--n, --d, and --v must all be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")


def _release(device: torch.device) -> None:
    """Release Python references and synchronize before a new measurement."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _memory_value(name: str, device: torch.device) -> int:
    """Read an allocator metric when the HCU PyTorch build exposes it."""
    function = getattr(torch.cuda, name, None)
    return int(function(device)) if function is not None else -1


def main() -> None:
    args = parse_args()
    _validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("HCU runtime is unavailable through torch.cuda")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)

    # All paths start from these exact same tensors.  They persist throughout
    # the benchmark and therefore are excluded from the incremental peak.
    torch.manual_seed(SEED)
    hidden_base = (torch.randn(args.n, args.d, device=device) * 0.02).to(torch.bfloat16)
    weight_base = (torch.randn(args.v, args.d, device=device) * 0.02).to(torch.bfloat16)
    labels = torch.arange(args.n, device=device, dtype=torch.long) % args.v
    labels[::17] = IGNORE_INDEX
    valid = labels != IGNORE_INDEX
    # -100 cannot index logits.  Replace ignored labels by zero only for gather;
    # the subsequent valid mask removes their contributions from the mean.
    safe_labels = labels.masked_fill(~valid, 0)

    def fusion_bf16() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Path 1: native Fusion through the public Megatron API."""
        hidden = hidden_base.detach().requires_grad_(True)
        weight = weight_base.detach().requires_grad_(True)
        loss = linear_cross_entropy(
            hidden,
            weight,
            labels,
            tp_group=None,
            reduction="mean",
            ignore_index=IGNORE_INDEX,
            sequence_parallel=False,
        )
        loss.backward()
        return loss, hidden.grad, weight.grad

    def unfused_bf16_linear_logsumexp() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Path 2: BF16 GEMM plus an explicit, correct CE formula.

        Materializing ``logits`` is the central memory difference from Fusion.
        ``float()`` makes the logsumexp reduction FP32 while leaving the linear
        projection in BF16, matching the usual mixed-precision model setup.
        """
        hidden = hidden_base.detach().requires_grad_(True)
        weight = weight_base.detach().requires_grad_(True)
        logits = F.linear(hidden, weight).float()
        token_nll = torch.logsumexp(logits, dim=-1) - logits.gather(
            1, safe_labels[:, None]
        ).squeeze(1)
        loss = token_nll[valid].mean()
        loss.backward()
        return loss, hidden.grad, weight.grad

    def unfused_fp32_linear_logsumexp() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Path 3: FP32 linear projection plus the same correct CE formula."""
        hidden = hidden_base.detach().requires_grad_(True)
        weight = weight_base.detach().requires_grad_(True)
        logits = F.linear(hidden.float(), weight.float())
        token_nll = torch.logsumexp(logits, dim=-1) - logits.gather(
            1, safe_labels[:, None]
        ).squeeze(1)
        loss = token_nll[valid].mean()
        loss.backward()
        return loss, hidden.grad, weight.grad

    def pytorch_bf16_linear_cross_entropy() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Path 4: ordinary BF16 F.linear followed by HCU cross-entropy.

        This is the standard mixed-precision formulation that most directly
        corresponds to Fusion's BF16 inputs and is a valid end-to-end
        standard-PyTorch performance baseline on the target HCU environment.
        """
        hidden = hidden_base.detach().requires_grad_(True)
        weight = weight_base.detach().requires_grad_(True)
        logits = F.linear(hidden, weight)
        loss = F.cross_entropy(logits, labels, reduction="mean", ignore_index=IGNORE_INDEX)
        loss.backward()
        return loss, hidden.grad, weight.grad

    def pytorch_fp32_linear_cross_entropy() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Path 5: the conventional FP32 PyTorch expression.

        This path is a valid FP32 baseline for the conventional materialized
        logits formulation.
        """
        hidden = hidden_base.detach().requires_grad_(True)
        weight = weight_base.detach().requires_grad_(True)
        logits = F.linear(hidden.float(), weight.float())
        loss = F.cross_entropy(logits, labels, reduction="mean", ignore_index=IGNORE_INDEX)
        loss.backward()
        return loss, hidden.grad, weight.grad

    def measure(
        name: str, function: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> BenchmarkResult:
        """Warm a path, measure one memory peak, then measure steady-state time."""
        _release(device)
        persistent_input_bytes = _memory_value("memory_allocated", device)

        # Native library load, plan construction, and allocator start-up occur
        # here rather than contaminating the timed steady-state samples.
        for _ in range(args.warmup):
            outputs = function()
            torch.cuda.synchronize(device)
            del outputs
            _release(device)

        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = _memory_value("memory_allocated", device)
        reserved_before = _memory_value("memory_reserved", device)
        outputs = function()
        torch.cuda.synchronize(device)
        peak_allocated = _memory_value("max_memory_allocated", device)
        peak_reserved = _memory_value("max_memory_reserved", device)
        loss_value = outputs[0].item()
        dhidden_finite = bool(torch.isfinite(outputs[1].float()).all().item())
        dweight_finite = bool(torch.isfinite(outputs[2].float()).all().item())
        del outputs
        _release(device)

        if not math.isfinite(loss_value):
            raise RuntimeError(f"{name} produced a non-finite loss: {loss_value}")
        if not dhidden_finite or not dweight_finite:
            raise RuntimeError(
                f"{name} produced non-finite gradients: "
                f"dhidden_finite={dhidden_finite}, dweight_finite={dweight_finite}"
            )

        # Reuse one pair of Events because each iteration waits for end_event
        # before recording it again.  Event time is the HCU-stream interval;
        # host time additionally contains Python/autograd dispatch and launch
        # overhead between the two perf_counter calls.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        device_samples_ms: list[float] = []
        host_samples_ms: list[float] = []
        for _ in range(args.iterations):
            host_start = time.perf_counter()
            start_event.record()
            outputs = function()
            end_event.record()
            # Waiting for this event establishes completion of the work between
            # the two records without adding a second timing mechanism.
            end_event.synchronize()
            device_samples_ms.append(start_event.elapsed_time(end_event))
            host_samples_ms.append((time.perf_counter() - host_start) * 1000.0)
            del outputs
            _release(device)

        return BenchmarkResult(
            name=name,
            loss=loss_value,
            dhidden_finite=dhidden_finite,
            dweight_finite=dweight_finite,
            device_median_ms=statistics.median(device_samples_ms),
            host_median_ms=statistics.median(host_samples_ms),
            host_mean_ms=statistics.mean(host_samples_ms),
            host_min_ms=min(host_samples_ms),
            host_max_ms=max(host_samples_ms),
            extra_peak_allocated_mib=(peak_allocated - allocated_before) / 1024**2,
            extra_peak_reserved_mib=(peak_reserved - reserved_before) / 1024**2,
            persistent_input_mib=persistent_input_bytes / 1024**2,
            device_samples_ms=device_samples_ms,
            host_samples_ms=host_samples_ms,
        )

    # Keep this order: Fusion first ensures its .so loading is warmed before
    # its measured samples, and FP32 logsumexp is available as the reference.
    benchmark_paths = (
        ("fusion_bf16", fusion_bf16),
        ("unfused_bf16_linear_logsumexp", unfused_bf16_linear_logsumexp),
        ("unfused_fp32_linear_logsumexp", unfused_fp32_linear_logsumexp),
        (
            "pytorch_bf16_linear_cross_entropy",
            pytorch_bf16_linear_cross_entropy,
        ),
        (
            "pytorch_fp32_linear_cross_entropy",
            pytorch_fp32_linear_cross_entropy,
        ),
    )
    results = [measure(name, function) for name, function in benchmark_paths]

    fp32_reference = next(
        result for result in results if result.name == "unfused_fp32_linear_logsumexp"
    )
    fusion = results[0]

    print("HCU_LINEAR_CE_BENCHMARK")
    print(
        f"device={torch.cuda.get_device_name(args.device)!r} "
        f"arch={getattr(torch.cuda.get_device_properties(args.device), 'gcnArchName', '')} "
        f"N={args.n} D={args.d} V={args.v} warmup={args.warmup} iterations={args.iterations}"
    )
    print(
        "name | loss | device_median_ms | host_median_ms | host_mean_ms | "
        "host_min_ms | host_max_ms | "
        "extra_peak_allocated_mib | extra_peak_reserved_mib | loss_delta_vs_fp32_reference"
    )
    for result in results:
        print(
            f"{result.name} | {result.loss:.9f} | {result.device_median_ms:.3f} | "
            f"{result.host_median_ms:.3f} | {result.host_mean_ms:.3f} | "
            f"{result.host_min_ms:.3f} | {result.host_max_ms:.3f} | "
            f"{result.extra_peak_allocated_mib:.1f} | "
            f"{result.extra_peak_reserved_mib:.1f} | "
            f"{abs(result.loss - fp32_reference.loss):.9g}"
        )

    pytorch_bf16 = next(
        result for result in results if result.name == "pytorch_bf16_linear_cross_entropy"
    )
    print(
        "fusion_vs_pytorch_bf16_cross_entropy_device: speedup={:.3f}x time_reduction={:.1f}% "
        "allocated_peak_reduction={:.1f} MiB ({:.1f}%)".format(
            pytorch_bf16.device_median_ms / fusion.device_median_ms,
            (
                1.0
                - fusion.device_median_ms / pytorch_bf16.device_median_ms
            )
            * 100.0,
            pytorch_bf16.extra_peak_allocated_mib - fusion.extra_peak_allocated_mib,
            (
                1.0
                - fusion.extra_peak_allocated_mib / pytorch_bf16.extra_peak_allocated_mib
            )
            * 100.0,
        )
    )
    print(
        "fusion_vs_pytorch_bf16_cross_entropy_host: speedup={:.3f}x time_reduction={:.1f}%".format(
            pytorch_bf16.host_median_ms / fusion.host_median_ms,
            (1.0 - fusion.host_median_ms / pytorch_bf16.host_median_ms) * 100.0,
        )
    )

if __name__ == "__main__":
    main()
