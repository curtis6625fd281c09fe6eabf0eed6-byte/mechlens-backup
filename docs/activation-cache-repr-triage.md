# ActivationCache representation: issue #1126 triage

**Status:** Investigation open · No library behavior change proposed yet  
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
- [Earlier diagnostic notes](../debugging/activation_cache_repr_1126.md) contain
  isolated CPU measurements. They are exploratory evidence, not measurements of
  the reported notebook stall. Their validation notes describe an earlier
  environment; the later suite results are summarized below.
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

**All result slots below are intentionally pending.** Fill them from a new,
identified reproduction; do not substitute the earlier exploratory timings.

### Reproduction identity

| Field | Value |
| --- | --- |
| Confirmed issue URL and exact failing cell | **Pending** |
| Checkout commit and imported class file | **Pending** |
| Python, TransformerLens, PyTorch, and IPython versions | **Pending** |
| Notebook frontend, version, and display extensions | **Pending** |
| Device, available memory, and tensor dtype/shapes | **Pending** |
| Key count and total key-text length | **Pending** |
| Logical tensor bytes and shared-storage/alias details | **Pending** |
| Tensor print options, including threshold and edge items | **Pending** |
| Warm-up, repetitions, and synchronization method | **Pending** |

### Results to collect

| Measurement | Median / spread | Output size | Observation / evidence |
| --- | --- | --- | --- |
| Model execution and cache construction, output suppressed | **Pending** | N/A | **Unfilled** |
| Initial outstanding CUDA work, if applicable | **Pending** | N/A | **Unfilled** |
| `repr(cache)` | **Pending** | **Pending** | **Unfilled** |
| `print(cache)` captured to memory | **Pending** | **Pending** | **Unfilled** |
| IPython formatting of the wrapper | **Pending** | **Pending** | **Unfilled** |
| Captured `display(cache)` publication | **Pending** | **Pending** | **Unfilled** |
| Actual notebook publication and frontend rendering | **Pending** | **Pending** | **Unfilled** |
| Underlying dictionary representation | **Pending** | **Pending** | **Unfilled** |
| One tensor and the output/cache tuple, separately | **Pending** | **Pending** | **Unfilled** |
| Fixed keys, increasing tensor storage | **Pending** | **Pending** | **Unfilled** |
| Fixed tensor values, increasing key count | **Pending** | **Pending** | **Unfilled** |
| Default versus full tensor-print profile, bounded inputs | **Pending** | **Pending** | **Unfilled** |

Record failures or hangs with a traceback or stack sample and elapsed time; do
not record them as zero-duration results. Mark a row **N/A** only after confirming
why it does not apply, such as running on CPU rather than CUDA.

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
> Our next step is to time cache construction, `repr(cache)`, IPython formatting,
> and actual notebook display separately, then vary key count independently of
> tensor size. We would like to establish which path is slow before changing
> the representation's behavior. A bounded key summary may be useful, but we
> have not established that it would fix this report.
