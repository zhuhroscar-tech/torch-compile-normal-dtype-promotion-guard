[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-compile-normal-dtype-promotion-guard

Detect and guard a real `torch.compile` correctness bug:
`torch.distributions.Normal.sample()` **silently promotes the output
dtype** under `torch.compile` in cases where eager mode **preserves**
the lower-precision `loc` dtype.

```python
loc = torch.tensor([0.0, 1.0], dtype=torch.float16)
scale = torch.tensor([1.0, 2.0], dtype=torch.float32)

def fn(loc, scale):
    return torch.distributions.Normal(loc, scale).sample()

fn(loc, scale).dtype                       # torch.float16 (eager preserves loc's dtype)
torch.compile(fn, fullgraph=True)(loc, scale).dtype   # torch.float32 (SILENTLY promoted!)
```

No error, no warning — just a silently different dtype flowing into
whatever consumes the sample next. If that consumer assumes a
fp16/bf16-sized buffer (e.g. a memory budget calculation, a
mixed-precision training step, or a downstream cast that now becomes a
narrowing cast instead of a no-op), the divergence can propagate
quietly.

Reproduced on this host (torch 2.14.0, CPU) across a 4-pair dtype
matrix:

| `loc` dtype | `scale` dtype | Eager result dtype | Compiled result dtype | Diverges? |
|---|---|---|---|---|
| float16  | float32 | float16  | float32 | **yes** |
| bfloat16 | float32 | bfloat16 | float32 | **yes** |
| float32  | float16 | float32  | float32 | no (eager already promotes here) |
| float64  | float32 | float64  | float64 | no |

The divergence is specific to a **lower-precision `loc` paired with a
higher-precision `scale`** — exactly the case where eager's own
dtype-preservation contract keeps `loc`'s dtype, but `torch.compile`'s
decomposition of `Normal.sample()` does not.

Upstream: [pytorch/pytorch#194547](https://github.com/pytorch/pytorch/issues/194547)
("torch.compile silently promotes dtype for
torch.distributions.Normal.sample() where eager preserves it"). Open
and unresolved as of this tool's last verification. Disclosure check:
searched English, Chinese (简体中文), and Japanese developer
communities for independent discussion of this specific issue; found
only the original English-language upstream report, no additional
native-language discussion surfaced.

## Install and diagnose

Requires Python 3.9+ and a PyTorch build. From source:

```bash
git clone https://github.com/zhuhroscar-tech/torch-compile-normal-dtype-promotion-guard.git
cd torch-compile-normal-dtype-promotion-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-compile-normal-dtype-promotion-guard
torch-compile-normal-dtype-promotion-guard --json
```

If you already manage a compatible PyTorch installation, install `.`
without the extra. Use `--no-color` for plain text output.

Exit codes: **0** means the guard restored eager's dtype-preservation
contract for every test case (and prints an "info" line, not a
warning, if the underlying bug also happened not to reproduce on your
build — e.g. after an upstream fix lands), **1** means the guard
failed to restore the dtype for at least one case, and **2** means
PyTorch could not be imported. A successful guard check does not by
itself mean the upstream bug was reproduced on your build; inspect
`any_native_divergence` separately.

## Python API

Wrap a `torch.compile`-produced callable together with its
(uncompiled) eager equivalent:

```python
from torch_compile_normal_dtype_promotion_guard import safe_compiled_normal_sample

compiled_fn = torch.compile(fn, fullgraph=True)
safe_fn = safe_compiled_normal_sample(compiled_fn, fn)

out = safe_fn(loc, scale)   # dtype matches eager (loc's dtype preserved)
```

**This is a cheap fix, unlike a value-corruption guard.** Because the
sampled *values* here are numerically fine (a promoted-precision draw
from the correct distribution) — only the *dtype* is wrong — the
wrapper does not need to recompute eagerly. It runs the compiled
function once and casts the result down to `loc`'s dtype only if the
dtype actually diverged, preserving `torch.compile`'s speed benefit in
both the common case (no divergence) and the divergent case (one cheap
cast).

## Scope and limitations

- Reproduced and tested on CPU only (torch 2.14.0, macOS arm64 +
  Ubuntu CI). Not separately verified on CUDA/MPS.
- Only guards `torch.distributions.Normal.sample()`'s dtype contract
  specifically; does not attempt to guard other `torch.distributions`
  classes, which were not verified to exhibit the same divergence.
- `diagnose()` always re-runs the actual reproduction against whatever
  torch build is installed — it never assumes a specific PyTorch
  version is or isn't affected. If the upstream fix lands and ships,
  `any_native_divergence` will correctly report `false`.
- Does not attempt to patch or monkeypatch `torch.distributions` or
  Dynamo internals — it is a call-boundary wrapper only, safe to use
  regardless of which torch version is installed.

## Development

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest --cov=torch_compile_normal_dtype_promotion_guard --cov-report=term-missing
```

## License

MIT
