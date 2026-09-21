# torch-compile-normal-dtype-promotion-guard

This guard has moved into the consolidated package:

**https://github.com/zhuhroscar-tech/torch-correctness-guards**

Use the umbrella package instead:

```bash
python -m pip install 'git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git[torch]'
torch-guard run normal-dtype-promotion
```

Python API:

```python
from torch_correctness_guards import safe_compiled_normal_sample
```

This repository is archived as part of the consolidation from many single-purpose PyTorch guard repos into one maintained package. The original functionality is preserved as `torch_correctness_guards.guards.normal_dtype_promotion` and the `torch-guard run normal-dtype-promotion` subcommand.

License: MIT.
