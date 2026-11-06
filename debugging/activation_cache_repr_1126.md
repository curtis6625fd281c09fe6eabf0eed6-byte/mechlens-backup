# ActivationCache representation investigation

Diagnostic only; no library behavior changes. The investigation name retains the
reported issue number, but the described report has not been verified: this
backup's issue #1126 returned 404, and upstream TransformerLens #1126 is an
unrelated attention-cleanup PR.

## Run the probe

From a checkout with its normal dependencies installed (`uv sync`):

```bash
uv run python -m debugging.activation_cache_repr_probe
uv run python -m debugging.activation_cache_repr_probe --include-values
uv run python -m debugging.activation_cache_repr_probe --keys 4 --elements 4096 --include-values --profile full
uv run python -m debugging.activation_cache_repr_probe --keys 100 --elements 256 --include-values
uv run python -m debugging.activation_cache_repr_probe --keys 100 --elements 163840 --include-values
```

The CLI defaults to CPU, eight independent tensors, and 4,096 elements each. No
model or weights are downloaded. Use `--device cuda` to investigate synchronization
on a CUDA machine. Full tensor printing is explicitly opt-in; start with small
inputs. Timings include a warm-up followed by five samples by default.

In the original notebook, finish cache construction in a separate cell, then:

```python
from debugging.activation_cache_repr_probe import probe_cache
report = probe_cache(cache)  # Existing kernel's formatter and print settings.
print(report)
# After inspecting that result, explicitly investigate the values:
value_report = probe_cache(cache, include_values=True)
print(value_report)
```

The report includes the actual class file and representation source, versions,
tensor devices, logical bytes, print options, and per-path timings. Logical bytes
can double-count shared storage. The probe repository commit does not establish
the version of an independently installed TransformerLens; check the class path.
The tensor print-options diagnostic uses PyTorch's private `_tensor_str.PRINT_OPTS`
because there is no public getter; this may need adapting for other versions.

`print` writes to memory; `display` publication is captured. Neither measures
notebook transport or browser rendering. IPython may catch formatter exceptions
internally, so inspect warnings/tracebacks as well as the report. CUDA devices in
the cache are synchronized outside the start of each sample and at completion;
initial outstanding CUDA work is timed separately. MPS is not synchronized.

## Evidence from the pinned source

Base: `33387b7def1ead3a8cba532a729da5599e875873` in this repository.

- `ActivationCache.__repr__` formats `list(self.cache_dict.keys())` only. There is
  no custom `__str__`, `_repr_pretty_`, or rich-display hook in the class.
- Normal print/str/repr and IPython's fallback use that representation. Registered
  formatters or notebook extensions can change the IPython path.
- `cache.cache_dict`, `cache[key]`, and `(output_tensor, cache)` reach tensor
  representation. Both HookedTransformer and TransformerBridge return an
  output/cache tuple; `return_cache_object=False` returns a dictionary as its
  cache component.
- Cache collection occurs during execution. HookedRootModule's saving hook uses
  `tensor.detach().to(device)` before storing the activation; that is not work
  performed by `ActivationCache.__repr__`.

## Isolated reproduction already performed

Python 3.12.14, IPython 9.16.1, PyTorch 2.15.0+cpu; no CUDA. The full class AST was
loaded from the pinned source without importing the package's unrelated model
dependencies. This is a representation-path experiment, not a full package or
original-notebook reproduction. Seven samples after warm-up, one PyTorch CPU
thread:

| 100 independent tensors | Wrapper repr | IPython wrapper formatting | Dictionary repr |
| --- | ---: | ---: | ---: |
| 100 KiB total | 0.007 ms | 0.049 ms | 134.849 ms |
| 62.5 MiB total | 0.008 ms | 0.050 ms | 10.605 ms |

The larger tensors were summarized under default print options; the smaller
tensors were printed in full. For four 4,096-element tensors, dictionary repr
increased from 0.391 ms (default) to 80.425 ms (full profile); wrapper repr stayed
at approximately 0.001 ms. These are illustrative measurements, not thresholds.

With Tensor.__repr__ patched to raise, wrapper repr/str/print, IPython formatting,
and captured display made zero calls to it. The dictionary, individual tensor,
and tensor/cache tuple controls reached it.

An earlier keys-only experiment produced about 3.09 million characters for
100,000 keys. Key count can therefore create substantial frontend output even
when tensors are never formatted.

## What to reproduce before a behavior change

1. Establish the exact displayed object, imported class, and formatter in the
   failing kernel. A dictionary or entire return tuple is a different path.
2. Separate model/cache construction, initial pending CUDA work, representation,
   formatter work, and actual publication/rendering.
3. Hold key count constant while varying tensor storage; then vary key count
   independently. Compare tensor print profiles on bounded inputs.
4. Only after formatting is isolated, compare actual `display(cache)` in the
   failing frontend. Captured display cannot reproduce browser stalls.

No observed result establishes the reported notebook hang's cause. A keys-only
summary limit could reduce output size, but it would not fix tensor formatting
that happens outside the wrapper.

## Probe validation

The probe was exercised against the extracted pinned class and real CPU tensors:
all nine formatting cases, a tensor-repr trap on wrapper-only cases, an empty
cache, invalid repetition counts, JSON serialization, and CLI default/full
profiles with print-options restoration. Full-profile reports encode infinite
thresholds as strings so the output remains valid JSON.

Formatting and focused type checks were run. Repository-wide formatting also
rewrote unrelated files with the available formatter version; those changes were
discarded. `make test-pr` could not start tests because this isolated environment
lacks pytest-doctestplus, jaxtyping, nbval, and rerun plugins. Repository-wide
mypy also reported shape-annotation/type errors in model code in this incomplete
dependency environment. These are unresolved validation limitations, not a claim
that the repository suite passes or that its errors are harmless. A full package
import and original notebook/GPU reproduction remain outstanding.
