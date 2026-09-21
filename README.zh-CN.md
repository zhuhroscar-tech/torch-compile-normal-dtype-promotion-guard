# torch-compile-normal-dtype-promotion-guard

此防护工具已迁移到统一维护的合并包：

**https://github.com/zhuhroscar-tech/torch-correctness-guards**

请改用 umbrella package：

```bash
python -m pip install 'git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git[torch]'
torch-guard run normal-dtype-promotion
```

Python API：

```python
from torch_correctness_guards import safe_compiled_normal_sample
```

作为 PyTorch 单用途 guard 仓库整合的一部分，本仓库已归档。原功能已保留在 `torch_correctness_guards.guards.normal_dtype_promotion` 模块和 `torch-guard run normal-dtype-promotion` 子命令中。

许可证：MIT。
