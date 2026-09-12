# HCU Linear CE independent feature

[中文副本与训练调用说明](linear_ce_adaptor_zh.md)

The `feature/hcu-linear-ce-adaptor` branch starts at `pr31-linear-ce`.
It reuses `linear_cross_entropy_for_training` from `Fusion_linear_ce`, without
changing the shared HCU GPT implementation or the upstream submodules.

## Integration

`LinearCrossEntropyFeature` owns `--use-hcu-linear-cross-entropy`, configuration
validation and patch registration. When enabled, the patch manager wraps the
existing HCU `GPTModel._postprocess` replacement. The wrapper validates inputs
before postprocessing and injects `linear_ce_output_processor` into the existing
`output_processor` hook. The processor uses the shared embedding weight when
provided, otherwise the output-layer weight, and bypasses vocabulary logits.

Without the flag no Linear CE wrapper is registered. Non-output pipeline stages,
unlabeled calls and active inference retain the original path. Labeled evaluation
uses fusion too. An existing output processor is a conflict, not silently replaced.

The configuration requires BF16, TP=1, CP=1, no SP, no MTP, no MuP, no vocabulary
parallel feature and no deferred embedding weight-gradient computation. Packed
sequences are unsupported; a binary loss mask is required. Start validation with
PP=1; pipeline combinations are not validated by this change.

The native operator only produces mean CE. The training adapter expands a scaled
mean to preserve the current masked-sum scheduler loss and gradients. These are
not actual per-token losses and must not be reused for per-token metrics or
arbitrary weighted losses. TP/CP support requires further native/adapter work.

## Run

Use the normal training environment, architecture-matched native artifact, data
and model arguments. Add:

```bash
--use-hcu-linear-cross-entropy --bf16 \
--tensor-model-parallel-size 1 --context-parallel-size 1 \
--pipeline-model-parallel-size 1
```

The flag is registered by the Feature; do not add it again in training/arguments.py.
Importing `hcu_megatron.megatron_adaptor` installs features, as in pretrain_gpt.py.

## Validation

From the worktree root (keep temporary writes inside this directory):

```bash
mkdir -p .test-tmp
TMPDIR="$PWD/.test-tmp" PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest discover -s tests/unit_tests/fusions \
  -p 'test_hcu_linear_cross_entropy*.py' -v
```

The adaptor tests use the real patch manager and the unchanged HCU postprocess
function body with substituted external dependencies. They do not load a complete
Megatron model. Numerical tests require PyTorch and substitute ordinary CPU CE
for the native kernel, comparing masked loss, hidden/weight gradients and gradient
accumulation; this does not validate native HCU execution.

Before production use, compare flag-off and flag-on HCU runs using identical
weights/batches, including shared embeddings, all-masked batches, gradient
accumulation and optimizer updates. Check native kernel invocation and memory use.

Container `yg_linear` result: all 18 contract/numerical tests passed with
PyTorch 2.10.0 (8 visible HCU devices). Full-model training remains unverified.
Megatron-LM and Bridge were initialized from local repositories at the recorded
commits; Energon and recursive dependencies may still need initialization in the
training environment.

Native public-API smoke test also passed in `yg_linear`: device `BW`,
architecture `gfx936:sramecc+:xnack-`, N=2048, D=4096, V=32000,
loss=10.372614860534668. It checks two forward/backward calls for finiteness
and repeatability, not reference accuracy or full-model optimizer updates.

Pre-commit review added an explicit empty-token input error and regression test.
Import ordering was checked with `uv run isort`; new Python modules were formatted
with Black. GPT runtime integration stays separate from Feature registration.
