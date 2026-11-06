# ActivationCache representation: issue #1126 triage

**Status:** Formatting matrix measured · Live notebook rendering blocked · No behavior change proposed

**Branch:** `investigate/activation-cache-repr-1126`

**Source baseline:** `33387b7def1ead3a8cba532a729da5599e875873`

## Summary

The inspected `ActivationCache` represents its **key names**, not its cached
tensors. A notebook stall attributed to “printing a large cache” therefore needs
an exact reproduction before changing `ActivationCache.py`. Displaying a tensor,
the underlying dictionary, or the model's entire return tuple takes a different
path and can invoke tensor formatting.

**Issue identity is unconfirmed.** The connected backup repository returned 404
for #1126; upstream TransformerLens #1126 is an unrelated attention-cleanup PR.
The issue number is retained as the investigation label. Confirm the intended
issue URL before posting the proposed response below.

## What the code establishes

[`ActivationCache.__repr__`](../transformer_lens/ActivationCache.py) returns:

```python
return f"ActivationCache with keys {list(self.cache_dict.keys())}"
```

The class defines no custom `__str__`, `_repr_pretty_`, or rich-display hook.

| Operation | Expected formatting path | Formats cached tensor values? |
| --- | --- | --- |
| `repr(cache)` | `ActivationCache.__repr__` → key list | No |
| `str(cache)` or `print(cache)` | Inherited string fallback → `__repr__` | No |
| Notebook `cache` or `display(cache)` | Normal IPython fallback → `__repr__` | No, unless a custom formatter changes the path |
| `repr(cache.cache_dict)` | Dictionary → each value's representation | Yes |
| Displaying `cache[key]` | Lookup → tensor representation | Yes, for that tensor |
| Displaying `(output_tensor, cache)` | Tuple → tensor and wrapper representations | Yes, for the output tensor |

Both [HookedTransformer](../transformer_lens/HookedTransformer.py) and
[TransformerBridge](../transformer_lens/model_bridge/bridge.py) return an
output/cache tuple from `run_with_cache`. With `return_cache_object=False`, its
cache component is a plain dictionary.

Cache collection and any requested device transfers happen during execution;
they are not triggered by the wrapper's representation. The wrapper's formatting
cost instead scales with key count and total key text. A very large key list can
still produce substantial notebook output.

## Evidence and limits

- [Regression tests](../tests/unit/test_activation_cache_repr.py) verify that
  single and multiple cache names appear and that cached tensors' `__repr__`
  methods are not invoked. All three cases passed with the locked dependencies.
- The bounded experiment below uses the real package imported from this checkout
  and locked dependency versions. The notebook is ready to run, but a live kernel
  and browser connection were blocked here. Earlier exploratory findings are
  consolidated in the historical section below.
- The initial broad unit run reported 3,008 passes and 18 failures. All 18
  failures passed on targeted reruns after aligning dependencies with the
  lockfile and installing the multimodal packages. This was not one clean full
  run in a single environment.
- The integration run reported 313 passes before worker crashes and an xdist
  internal error. Two affected tests passed sequentially; the Gemma-2-2B test
  then returned HTTP 401 for gated model access. Acceptance, benchmark,
  docstring, and notebook stages were not run.
- These results do not establish the original stall's cause, frontend rendering
  performance, or CUDA behavior. They do not justify saying the full suite passes.

## Measurements

### Scope and execution status

These are **synthetic CPU formatting measurements**, not a reproduction of the
reported hang. The planned notebook kernel died before execution with
`Operation not permitted (src/ip_resolver.cpp:542)` and
`Kernel died before replying to kernel_info`. JupyterLab's server started, but
the cloud browser returned `ERR_CONNECTION_REFUSED` for its localhost URL.
The server was stopped after the attempt.

The formatting cells were therefore run in **in-process IPython**, including
captured `display()` for the wrapper, dictionary, and a tensor. No live IOPub
publication, browser rendering, or visible responsiveness was measured. The
notebook has no fabricated execution counts or outputs.

- [Runnable notebook](../debugging/activation_cache_repr_experiment.ipynb)
- [Raw results and all samples](../debugging/activation_cache_repr_measurements.json)
- [Diagnostic probe](../debugging/activation_cache_repr_probe.py)

### Reproduction identity

| Field | Recorded value |
| --- | --- |
| Source commit | `8a410e9c35666bdb0ecec3dfb466442408a3a879` |
| ActivationCache source blob | `7d392b6da53f44bc975d13606769a20682088dc9` |
| Import | Real `transformer_lens.ActivationCache.ActivationCache` from this checkout; exact file path in JSON |
| Versions | Python 3.12.14; TransformerLens 0.0.0; PyTorch 2.10.0+cpu; IPython 9.10.0 |
| UI attempt | JupyterLab 3.6.8 server; browser could not connect; kernel startup blocked |
| Data | CPU float32, independently allocated all-ones vectors; no shared-storage aliases |
| Allocation bounds | At most 16 MiB tensor storage; 10,000 keys; full profile limited to four 4,096-element tensors |
| Print options | Precision 4, edge items 3, line width 80; threshold 1,000 (default) or infinity (full) |
| Timing | One warm-up, five samples, median and observed min–max; one PyTorch thread |
| Construction | Separately timed once per case, without warm-up; excluded from formatting measurements |
| CUDA/MPS synchronization | N/A: CPU-only experiment |
| Issue URL and original failing cell | **Pending**; issue identity investigation deferred |

The exact UTC timestamp and unrounded samples are in the JSON. The small sample
count and shared host make these illustrative measurements, not performance
thresholds or confidence intervals.

### Key count, fixed one-element tensors

Only keys change; tensor values have constant shape. Timings are milliseconds.

| Keys | Wrapper repr characters | `repr(cache)` median | IPython formatting median | Captured wrapper display median |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 53 | 0.0009 | 0.0916 | 0.2900 |
| 100 | 2,816 | 0.0085 | 0.1024 | 0.1467 |
| 10,000 | 298,916 | 0.7919 | 1.0531 | 1.4960 |

### Tensor size and print profile, fixed four keys

Wrapper representations are 134 characters in every row. Timings are milliseconds.

| Elements per tensor | Total tensor bytes | Profile | Wrapper repr median | Dictionary repr median | Dictionary repr characters |
| ---: | ---: | --- | ---: | ---: | ---: |
| 256 | 4,096 | default | 0.0008 | 5.7830 | 4,692 |
| 4,096 | 65,536 | default | 0.0009 | 0.4358 | 268 |
| 1,048,576 | 16,777,216 | default | 0.0011 | 0.4460 | 268 |
| 4,096 | 65,536 | full | 0.0008 | 80.7704 | 72,948 |

Larger tensors can format faster under the default profile because they are
summarized. Changing tensor storage alone did not increase wrapper output size.
Full printing affected dictionary/tensor formatting, not the wrapper's repr.

### Captured display comparisons

Four keys, 4,096 elements per tensor; a separate in-process IPython session in
the same locked environment. These include capture-context overhead, and exclude
real notebook transport and rendering. Output size is captured `text/plain`.

| Profile | Displayed object | Median (ms) | Min–max (ms) | Characters |
| --- | --- | ---: | ---: | ---: |
| default | wrapper | 0.1457 | 0.1252–0.2644 | 134 |
| default | dictionary | 0.8507 | 0.8214–0.9639 | 271 |
| default | tensor | 0.3213 | 0.3073–0.3576 | 38 |
| full | wrapper | 0.1391 | 0.1219–0.2460 | 134 |
| full | dictionary | 85.9160 | 85.0763–87.2706 | 73,859 |
| full | tensor | 23.1497 | 21.8586–28.7176 | 18,208 |

### Detailed baseline and remaining result slots

The measured baseline is four keys × 4,096 float32 elements under default printing.

| Measurement | Median (ms) | Min–max (ms) | Output characters / status |
| --- | ---: | ---: | --- |
| `repr(cache)` | 0.0009 | 0.0008–0.0030 | 134 |
| `print(cache)` to memory | 0.0017 | 0.0015–0.0027 | 135 |
| IPython wrapper formatting | 0.0900 | 0.0825–0.1358 | Not recorded by probe |
| Dictionary repr | 0.4358 | 0.4205–0.4832 | 268 |
| One tensor repr | 0.1131 | 0.1059–0.3616 | 38 |
| Tensor/cache tuple repr | 0.1900 | 0.1193–0.2417 | 176 |
| Model inference | N/A | N/A | Synthetic tensors; no model loaded |
| Initial CUDA work | N/A | N/A | CPU-only |
| Live notebook publication | **Pending** | **Pending** | Kernel startup blocked |
| Browser rendering / responsiveness | **Pending** | **Pending** | Frontend connection blocked |
| Original failing notebook | **Pending** | **Pending** | Not reproduced |

Construction timings are single samples in the raw results, not medians. Do not
replace blocked or pending results with zero or with captured-display timings.

### Run the notebook in a normal Jupyter environment

Use the repository's locked dependencies and register/select its kernel:

```bash
uv sync --frozen
uv run python -m ipykernel install --user --name repr-locked --display-name "Python (locked repr experiment)"
uv run jupyter lab debugging/activation_cache_repr_experiment.ipynb
```

Run all cells. The final cells publish bounded real outputs and heartbeat
messages. Observe whether scrolling and interaction remain responsive, recording
the frontend/browser versions. `display()` return time is only a kernel-side
measurement; a heartbeat alone does not measure browser paint completion.
The notebook restores print settings after each profile comparison.

## Historical exploratory evidence

Earlier experiments loaded the class AST in isolation on PyTorch 2.15.0+cpu and
IPython 9.16.1. With 100 fixed keys, wrapper repr medians stayed at 0.007–0.008 ms
between 100 KiB and 62.5 MiB of tensors; dictionary repr was 134.849 ms and
10.605 ms respectively because the larger tensors were summarized. Full printing
of four 4,096-element tensors increased dictionary repr from 0.391 to 80.425 ms.
These figures are historical and are not pooled with the current measurements.

Raising-value/model controls confirmed that wrapper print/str/repr, normal IPython
formatting, and captured display did not represent those objects. A separate
100,000-key experiment produced about 3.09 million characters, demonstrating why
key count is still relevant to frontend output volume. None reproduced a hang.

## Reproduction plan

1. Confirm the exact expression and imported class in the failing kernel. Check
   whether the object is a wrapper, dictionary, tensor, or entire return tuple.
2. Finish model execution and cache construction in a separate cell with output
   suppressed. Keep that time separate from representation and display.
3. Run the [diagnostic probe](../debugging/activation_cache_repr_probe.py) against
   the existing cache without publishing the large representation:

   ```python
   from debugging.activation_cache_repr_probe import probe_cache

   report = probe_cache(cache)
   print(report)
   ```

4. Explicitly opt into `probe_cache(cache, include_values=True)` on bounded
   inputs. Compare tensor storage size independently from key count and compare
   default/full print profiles. Large tensors are normally summarized; “all
   elements are printed” is not a safe assumption.
5. Measure actual notebook display separately. The probe captures publication
   and cannot measure browser rendering or notebook transport. It synchronizes
   CUDA devices found in cached tensors but does not synchronize MPS.

## Decision criteria

| Finding | Next action |
| --- | --- |
| Wrapper formatting is fast; dictionary or tensor formatting is slow | Investigate tensor formatting, print options, and the exact displayed object |
| Wrapper output grows excessively with key count | Evaluate a bounded key summary as a separate usability change |
| Formatting is fast; actual notebook display stalls | Investigate transport, frontend rendering, and custom formatters |
| Execution or initial CUDA synchronization dominates | Investigate model execution, caching, and device transfers |
| The failing kernel uses another implementation | Reproduce against that implementation before proposing a fix here |

Keep the keys-only regression tests regardless of whether a separate display
improvement is later justified. Do not describe a summary-format change as a fix
for the reported hang until the relevant path is reproduced.

## Proposed maintainer response for GitHub issue #1126

**Draft only — not posted. Confirm the intended issue URL first.**

> Thanks for reporting this. In the version we inspected, `ActivationCache`'s
> representation formats only the cached activation names; it does not format
> the tensor values. We added regression tests for those two guarantees, and
> they pass. This does not yet reproduce or explain the notebook stall.
>
> Could you share the exact failing cell, your TransformerLens/PyTorch/IPython
> versions, and the notebook frontend? It would help to distinguish displaying
> `cache` from `cache.cache_dict`, an individual tensor, or the complete
> `(output, cache)` result. Tensor print settings and custom display extensions
> may also matter.
>
> Bounded CPU measurements show that wrapper formatting tracks key count, while
> full tensor printing makes dictionary/tensor formatting more expensive. Live
> notebook publication and browser rendering remain unmeasured; our next step
> is to test those in the affected frontend. We would like to establish which
> path is slow before changing the representation's behavior. A bounded key summary may be useful, but we
> have not established that it would fix this report.
