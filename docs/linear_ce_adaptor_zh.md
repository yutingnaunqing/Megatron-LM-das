# HCU Linear CE 独立特性适配

本文是 `linear_ce_adaptor.md` 的中文副本，并补充正常训练的调用方法。

开发分支为 `feature/hcu-linear-ce-adaptor`，基于 `pr31-linear-ce`。
复用 `Fusion_linear_ce` 中的 `linear_cross_entropy_for_training`，不修改公共
HCU GPT 实现及上游子模块代码。

## 实现方式

`LinearCrossEntropyFeature` 管理开关 `--use-hcu-linear-cross-entropy`、配置校验和补丁注册。
启用时，补丁管理器包装已有的 HCU `GPTModel._postprocess`，先检查兼容性，
再通过原函数的 `output_processor` 接口调用 `linear_ce_output_processor`。
后者优先使用共享 embedding 权重，否则使用输出层权重，直接调用融合算子，避免生成完整词表 logits。

不开启开关时不注册该 wrapper。非输出流水线阶段、无 labels 的调用和活动推理模式
继续走原路径；带 labels 的验证阶段也使用融合。已有其他 `output_processor` 时明确报冲突。

## 使用限制

- 必须使用 BF16、TP=1、CP=1，关闭 sequence parallel。
- 不支持 MTP、MuP、`enable_vocab_parallel` 和延迟 embedding 权重梯度计算。
- 不支持 packed sequence；必须提供二值 `loss_mask`。
- 首次验证建议 PP=1；本改动尚未验证流水线组合。
- native 算子返回 mean CE；训练适配器展开经过缩放的均值，保持当前 masked-sum
  聚合下的总 loss 和梯度。这不是真实逐 token loss，不能用于逐 token 指标或任意加权 loss。
- TP/CP 支持需要另外扩展 native 算子与训练适配逻辑。

## 正常训练如何调用

不需要手动调用 Python 融合函数，也不需要另写训练循环。
使用本 worktree 的 `pretrain_gpt.py`，在原有训练参数中加入：

```bash
--use-hcu-linear-cross-entropy
--bf16
--tensor-model-parallel-size 1
--context-parallel-size 1
--pipeline-model-parallel-size 1
```

必须删除原有 `--sequence-parallel`，并去掉上述不支持的功能参数。
普通 CE fusion 开关不能代替 `--use-hcu-linear-cross-entropy`。
参数由 Feature 注册，无需在 `training/arguments.py` 再次添加。
`pretrain_gpt.py` 已导入 `hcu_megatron.megatron_adaptor`，会安装相应补丁。

### 使用仓库 Llama2 脚本

在新 worktree 中修改 `examples/llama2/train_llama2_7B.sh` 的配置：

1. 将 `TP=1`、`PP=1`、`CP=1`。
2. 在 `TRAINING_ARGS` 数组内加入 `--use-hcu-linear-cross-entropy`，保留 `--bf16`。
3. 从 `MODEL_PARALLEL_ARGS` 删除 `--sequence-parallel`。
4. 根据显存设置模型大小、序列长度、micro batch 和重计算。PP 从 2 改为 1 后，
   每卡容纳的模型参数增加，原 7B 配置不保证适合当前显存。
5. 在同目录 `run_llama.sh` 配置真实的 DTK 环境、数据、tokenizer、checkpoint 路径
   和 launcher，然后按原有方式启动。

例如，在路径和 hostfile 配好后：

```bash
cd /public/home/yugeng/work/Megatron-LM-das-linear-ce-adaptor/examples/llama2
bash run_llama.sh hostfile 1
```

注意：这两个 shell 脚本不通用透传任意训练参数。仅在 `bash run_llama.sh ...`
末尾追加融合开关不会自动传给 Python，应放进上述 `TRAINING_ARGS` 数组。
多卡 TP=PP=CP=1 时通常使用数据并行，不需要为 Linear CE 单独创建 TP 通信组。

### 确保加载新 worktree 和 native 库

在 `yg_linear` 容器中，worktree 使用与宿主机相同的路径：

```bash
cd /public/home/yugeng/work/Megatron-LM-das-linear-ce-adaptor
export PYTHONPATH="$PWD:$PWD/3rdparty/Megatron-LM:$PWD/3rdparty/Megatron-Bridge/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p .test-tmp .cache
export TMPDIR="$PWD/.test-tmp"
export HCU_LINEAR_CE_CACHE_DIR="$PWD/.cache/hcu_linear_ce"
export TRITON_HOME="$PWD/.cache/triton"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
```

使用与设备架构及 PyTorch 版本匹配的 native artifact。已安装正确的 artifact wheel
时可自动发现 `.so`；也可以设置 `HCU_LINEAR_CE_EXTENSION_PATH` 为实际库文件的绝对路径。
不要照抄不存在的库路径。用 `HCU_LINEAR_CE_LOG_LEVEL=1` 查看 native 执行/选优日志。
多进程启动时，需要把库路径、缓存路径等环境变量传给各训练进程。

## 验证方法

在 worktree 根目录执行：

```bash
mkdir -p .test-tmp
TMPDIR="$PWD/.test-tmp" PYTHONDONTWRITEBYTECODE=1 \
python -m unittest discover -s tests/unit_tests/fusions \
  -p 'test_hcu_linear_cross_entropy*.py' -v
```

适配契约测试使用真实补丁管理器及未改动的 HCU postprocess 函数体，并替换外部依赖；
不等价于构建完整 Megatron 模型。数值测试使用 PyTorch CPU CE 替代 native kernel，
对比 masked loss、hidden/weight 梯度和梯度累积，也不等价于验证 HCU kernel。

仓库另有 native 公共接口 smoke 测试：

```bash
python tests/unit_tests/fusions/validate_hcu_linear_cross_entropy.py
```

该脚本测试 native forward/backward，不是完整训练。
正式使用前，还应以相同权重和 batch 对比融合开关开/关的 HCU 训练，检查共享权重、
全 mask、梯度累积、优化器更新、native 调用和显存占用。

Megatron-LM 和 Bridge 已从本地仓库初始化到基线记录的提交；Energon 和递归依赖可能仍需
在训练环境中补齐。具体测试结果见下方记录。

## 本次容器测试记录

在 `yg_linear` 中，从新 worktree 执行：

- PyTorch：2.10.0；可见设备数：8。
- 18 项契约及 CPU 数值测试全部通过，无跳过。
- 实际 HCU native 公共接口 smoke 测试通过，输出 `MEGATRON_LINEAR_CE_SMOKE_PASS`。
- native 测试设备为 `BW`，架构 `gfx936:sramecc+:xnack-`；
  输入 N=2048、D=4096、V=32000，loss=10.372614860534668。
  测试连续执行两次 forward/backward，检查有限值和重复一致性。
- 本次未运行完整模型训练，未验证优化器更新或多卡训练；native smoke
  也不提供与独立参考实现的数值误差对比。不能把这些结果视为完整训练验收。

提交前复核补充了空 token 输入校验和回归测试，并执行 `uv run isort` 与新增模块的 Black 格式化。
独立的 `core/models/gpt/linear_cross_entropy.py` 在技术上可以合并到 Feature 文件，
但保留它可将运行时 GPT 逻辑与开关/补丁注册分离，便于单独测试和维护；不是算子本身的要求。
