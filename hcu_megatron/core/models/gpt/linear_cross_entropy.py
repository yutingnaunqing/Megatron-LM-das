# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Optional Linear CE integration through GPT's output processor."""

from functools import wraps
from inspect import signature
from typing import Any, Callable


def validate_linear_ce_config(config: Any) -> None:
    """Reject unsupported configurations before GPT postprocessing executes."""
    unsupported = []
    for name in ("tensor_model_parallel_size", "context_parallel_size"):
        if getattr(config, name, 1) != 1:
            unsupported.append(f"{name} must be 1")
    for name in (
        "sequence_parallel",
        "mtp_num_layers",
        "defer_embedding_wgrad_compute",
        "use_mup",
        "enable_vocab_parallel",
    ):
        if getattr(config, name, False):
            unsupported.append(f"{name} must be disabled")
    if unsupported:
        raise ValueError("HCU Linear CE does not support: " + "; ".join(unsupported))


def linear_ce_output_processor(
    *,
    hidden_states: Any,
    output_layer: Any,
    output_weight: Any,
    labels: Any,
    loss_mask: Any,
    **kwargs: Any,
) -> Any:
    """Compute the scheduler-compatible loss without materializing logits."""
    from hcu_megatron.core.fusions.fused_linear_cross_entropy import linear_cross_entropy_for_training

    weight = output_weight if output_weight is not None else output_layer.weight
    return linear_cross_entropy_for_training(hidden_states, weight, labels, loss_mask)


def linear_ce_postprocess_wrapper(original: Callable) -> Callable:
    """Inject an output processor into the existing HCU GPT implementation."""
    call_signature = signature(original)

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from megatron.core.inference.utils import InferenceMode

        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        model = values["self"]
        if not model.post_process or values["labels"] is None or InferenceMode.is_active():
            return original(*args, **kwargs)

        # Validate before original can run MTP or another output processor.
        validate_linear_ce_config(model.config)
        if values.get("mtp_in_postprocess"):
            raise ValueError("HCU Linear CE does not support mtp_in_postprocess")
        if values.get("packed_seq_params") is not None:
            raise ValueError("HCU Linear CE does not support packed_seq_params")
        if values.get("loss_mask") is None:
            raise ValueError("HCU Linear CE requires loss_mask")
        if values.get("output_processor") is not None:
            raise ValueError("HCU Linear CE conflicts with an existing output_processor")

        bound.arguments["output_processor"] = linear_ce_output_processor
        return original(*bound.args, **bound.kwargs)

    return wrapped
