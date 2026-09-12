# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Feature registration for the optional HCU Linear CE training path."""

from argparse import ArgumentParser, Namespace
from typing import Any

from ..feature import AbstractFeature


class LinearCrossEntropyFeature(AbstractFeature):
    """Enable fused output projection and CE through a GPT wrapper."""

    def __init__(self) -> None:
        super().__init__("use-hcu-linear-cross-entropy")

    def register_args(self, parser: ArgumentParser) -> None:
        group = parser.add_argument_group(self.feature_name)
        group.add_argument(
            "--use-hcu-linear-cross-entropy",
            action="store_true",
            help="Use HCU fused Linear CE (BF16, TP=1, CP=1, no SP/MTP, binary loss mask).",
        )

    def validate_args(self, args: Namespace) -> Namespace:
        if not getattr(args, self.feature_name, False):
            return args
        from hcu_megatron.core.models.gpt.linear_cross_entropy import validate_linear_ce_config

        validate_linear_ce_config(args)
        if not getattr(args, "bf16", False):
            raise ValueError("HCU Linear CE requires --bf16")
        return args

    def register_patches(self, patch_manager: Any, args: Namespace) -> None:
        from hcu_megatron.core.models.gpt.linear_cross_entropy import linear_ce_postprocess_wrapper

        patch_manager.register_patch(
            "megatron.core.models.gpt.gpt_model.GPTModel._postprocess",
            linear_ce_postprocess_wrapper,
            apply_wrapper=True,
        )
