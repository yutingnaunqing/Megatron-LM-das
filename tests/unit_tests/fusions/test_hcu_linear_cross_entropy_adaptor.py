# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests using the real patch manager and HCU postprocess body."""

import argparse
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load("linear_ce_adapter", "hcu_megatron/core/models/gpt/linear_cross_entropy.py")
patches = load("linear_ce_patches", "hcu_megatron/patch_utils.py")
base = load("linear_ce_features.feature", "hcu_megatron/features_manager/feature.py")
with mock.patch.dict(sys.modules, {"linear_ce_features.feature": base}):
    feature_module = load(
        "linear_ce_features.fusions.linear_cross_entropy_feature",
        "hcu_megatron/features_manager/fusions/linear_cross_entropy_feature.py",
    )


class TestLinearCEAdaptor(unittest.TestCase):
    def setUp(self):
        self.inference = SimpleNamespace(is_active=lambda: False)
        self.module_context = mock.patch.dict(
            sys.modules,
            {
                "megatron.core.inference.utils": SimpleNamespace(InferenceMode=self.inference),
                "hcu_megatron.core.models.gpt.linear_cross_entropy": adapter,
            },
        )
        self.module_context.start()
        self.addCleanup(self.module_context.stop)
        self.feature = feature_module.LinearCrossEntropyFeature()
        self.model = SimpleNamespace(
            compute_language_model_loss=object(),
            _scale_logits=object(),
            post_process=True,
            config=SimpleNamespace(mtp_num_layers=None),
            share_embeddings_and_output_weights=False,
            output_layer=SimpleNamespace(weight=object()),
        )
        self.calls = []

    def make_wrapper(self):
        # Use the production parameter name for the model.
        calls = self.calls

        def original(
            self,
            hidden_states,
            labels,
            loss_mask=None,
            *,
            output_processor=None,
            packed_seq_params=None,
            mtp_in_postprocess=None
        ):
            calls.append(output_processor)
            return output_processor

        return adapter.linear_ce_postprocess_wrapper(original)

    def test_args_are_opt_in_and_validate(self):
        parser = argparse.ArgumentParser()
        self.feature.register_args(parser)
        self.assertFalse(parser.parse_args([]).use_hcu_linear_cross_entropy)
        args = parser.parse_args(["--use-hcu-linear-cross-entropy"])
        with self.assertRaisesRegex(ValueError, "bf16"):
            self.feature.validate_args(args)
        args.bf16 = True
        self.assertIs(self.feature.validate_args(args), args)
        for name, value in [
            ("tensor_model_parallel_size", 2),
            ("context_parallel_size", 2),
            ("sequence_parallel", True),
            ("mtp_num_layers", 1),
            ("use_mup", True),
            ("enable_vocab_parallel", True),
            ("defer_embedding_wgrad_compute", True),
        ]:
            with self.subTest(name=name):
                setattr(args, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    self.feature.validate_args(args)
                delattr(args, name)

    def test_positional_keyword_and_eval_calls(self):
        wrapped = self.make_wrapper()
        self.model.training = False
        self.assertIs(wrapped(self.model, "hidden", "labels", "mask"), adapter.linear_ce_output_processor)
        self.assertIs(
            wrapped(self.model, hidden_states="hidden", labels="labels", loss_mask="mask"),
            adapter.linear_ce_output_processor,
        )

    def test_fallback_preserves_processor(self):
        wrapped = self.make_wrapper()
        sentinel = object()
        for stage, labels, inference in [(False, "labels", False), (True, None, False), (True, "labels", True)]:
            self.model.post_process = stage
            self.inference.is_active = lambda: inference
            self.assertIs(wrapped(self.model, "hidden", labels, output_processor=sentinel), sentinel)

    def test_rejections_happen_before_original(self):
        wrapped = self.make_wrapper()
        for extra, message in [
            ({"loss_mask": None}, "loss_mask"),
            ({"packed_seq_params": object()}, "packed_seq_params"),
            ({"output_processor": object()}, "output_processor"),
            ({"mtp_in_postprocess": True}, "mtp_in_postprocess"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                wrapped(self.model, "hidden", "labels", **({"loss_mask": "mask"} | extra))
        self.model.config.mtp_num_layers = 1
        with self.assertRaisesRegex(ValueError, "mtp_num_layers"):
            wrapped(self.model, "hidden", "labels", "mask")
        self.assertEqual(self.calls, [])

    def test_patch_manager_composes_with_real_hcu_body(self):
        # Execute the unchanged production body with only its external dependencies
        # substituted. This verifies the real output_processor location and signature.
        tree = ast.parse((ROOT / "hcu_megatron/core/models/gpt/gpt_model.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gpt_model_postprocess")
        namespace = {"InferenceMode": self.inference}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<hcu-postprocess>", "exec"), namespace)
        hcu_original = namespace["gpt_model_postprocess"]
        target = types.ModuleType("linear_ce_patch_target")
        target.GPTModel = type("GPTModel", (), {"_postprocess": lambda *a, **kw: "upstream"})
        manager = patches.MegatronPatchesManager
        old_registry = manager.patches_info
        manager.patches_info = {}
        self.addCleanup(setattr, manager, "patches_info", old_registry)
        with mock.patch.dict(sys.modules, {"linear_ce_patch_target": target}):
            path = "linear_ce_patch_target.GPTModel._postprocess"
            manager.register_patch(path, hcu_original)
            # Verify Feature registers the expected public target and wrapper.
            recorder = mock.Mock()
            self.feature.register_patches(recorder, argparse.Namespace())
            args, kwargs = recorder.register_patch.call_args
            self.assertEqual(args[0], "megatron.core.models.gpt.gpt_model.GPTModel._postprocess")
            manager.register_patch(path, args[1], **kwargs)
            manager.apply_patches()
            processor = mock.Mock(return_value="fused-loss")
            with mock.patch.object(adapter, "linear_ce_output_processor", processor):
                result = target.GPTModel._postprocess(
                    self.model,
                    "hidden",
                    "ids",
                    "positions",
                    "labels",
                    None,
                    None,
                    None,
                    loss_mask="mask",
                )
            self.assertEqual(result, "fused-loss")
            self.assertIs(processor.call_args.kwargs["output_layer"], self.model.output_layer)
            self.assertEqual(processor.call_args.kwargs["loss_mask"], "mask")
            manager.remove_patches()
            self.assertEqual(target.GPTModel._postprocess(), "upstream")

    def test_processor_selects_shared_or_output_weight(self):
        native = mock.Mock(return_value="loss")
        fake = SimpleNamespace(linear_cross_entropy_for_training=native)
        with mock.patch.dict(
            sys.modules,
            {
                "hcu_megatron.core.fusions.fused_linear_cross_entropy": fake,
            },
        ):
            for shared in (None, object()):
                adapter.linear_ce_output_processor(
                    hidden_states="hidden",
                    output_layer=self.model.output_layer,
                    output_weight=shared,
                    labels="labels",
                    loss_mask="mask",
                )
                self.assertIs(native.call_args.args[1], self.model.output_layer.weight if shared is None else shared)


if __name__ == "__main__":
    unittest.main()
