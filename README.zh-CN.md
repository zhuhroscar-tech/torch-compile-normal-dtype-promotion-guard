[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-compile-normal-dtype-promotion-guard

检测并防护一个真实的 `torch.compile` 正确性问题：在 eager 模式下
`torch.distributions.Normal.sample()` 会保留低精度 `loc` 的数据类型，
但在 `torch.compile` 下会**悄悄地**将输出数据类型提升为更高精度，
不报任何错误或警告。

```python
loc = torch.tensor([0.0, 1.0], dtype=torch.float16)
scale = torch.tensor([1.0, 2.0], dtype=torch.float32)

def fn(loc, scale):
    return torch.distributions.Normal(loc, scale).sample()

fn(loc, scale).dtype                       # torch.float16（eager 保留 loc 的类型）
torch.compile(fn, fullgraph=True)(loc, scale).dtype   # torch.float32（悄悄被提升！）
```

在本机（torch 2.14.0，CPU）上验证过 4 组数据类型组合：

| `loc` 类型 | `scale` 类型 | Eager 结果类型 | Compiled 结果类型 | 是否偏差 |
|---|---|---|---|---|
| float16  | float32 | float16  | float32 | **是** |
| bfloat16 | float32 | bfloat16 | float32 | **是** |
| float32  | float16 | float32  | float32 | 否（eager 本身就会提升） |
| float64  | float32 | float64  | float64 | 否 |

上游 issue：[pytorch/pytorch#194547](https://github.com/pytorch/pytorch/issues/194547)。
截至本工具最近一次验证，该问题仍处于开放（未修复）状态。

披露检查：已在英文、简体中文、日文开发者社区中搜索是否有独立讨论此问题；
仅发现原始英文上游 issue 报告，未发现额外的母语社区讨论。

## 安装与诊断

需要 Python 3.9+ 及 PyTorch：

```bash
git clone https://github.com/zhuhroscar-tech/torch-compile-normal-dtype-promotion-guard.git
cd torch-compile-normal-dtype-promotion-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-compile-normal-dtype-promotion-guard
torch-compile-normal-dtype-promotion-guard --json
```

退出码：**0** 表示防护逻辑在所有测试用例中都恢复了 eager 的数据类型保留契约，
**1** 表示至少一个用例未能恢复，**2** 表示无法导入 PyTorch。

## Python API

```python
from torch_compile_normal_dtype_promotion_guard import safe_compiled_normal_sample

compiled_fn = torch.compile(fn, fullgraph=True)
safe_fn = safe_compiled_normal_sample(compiled_fn, fn)

out = safe_fn(loc, scale)   # 类型与 eager 一致（保留 loc 的类型）
```

**这是一个低成本修复**，与需要值修正的防护工具不同：由于采样出来的数值本身
是正确的（只是数据类型不对），包装器无需重新以 eager 方式计算，只需在类型
确实发生偏差时做一次类型转换即可，因此在两种情况下都保留了 `torch.compile`
的速度优势。

## 范围与限制

- 仅在 CPU 上复现并测试过（torch 2.14.0，macOS arm64 + Ubuntu CI），
  未单独验证 CUDA/MPS。
- 仅针对 `torch.distributions.Normal.sample()` 的数据类型契约；未验证
  其他 `torch.distributions` 类是否存在相同问题。
- `diagnose()` 始终针对当前安装的 torch 版本重新运行真实复现，从不假设
  某个版本受影响或不受影响。

## 开发

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_compile_normal_dtype_promotion_guard --cov-report=term-missing
```

## 许可证

MIT
