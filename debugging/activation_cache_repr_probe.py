"""Separate ActivationCache formatting from value formatting and notebook rendering.

Run from the repository root with:
    uv run python -m debugging.activation_cache_repr_probe --include-values

In the failing notebook:
    from debugging.activation_cache_repr_probe import probe_cache
    report = probe_cache(cache)
    print(report)

No model download or forward pass is needed for the synthetic CLI probe.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import math
import platform
import statistics
import subprocess
import time
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import torch


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def probe_cache(cache: Any, repeats: int = 5, include_values: bool = False) -> dict:
    """Measure an existing cache without publishing its contents.

    Value formatting is opt-in: it can be expensive with full tensor printing.
    CUDA work is synchronized before and after each timed sample. The initial
    synchronization is reported separately. CPU/MPS/frontend results are not
    interchangeable; MPS synchronization is not implemented here.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    tensors = [v for v in cache.cache_dict.values() if isinstance(v, torch.Tensor)]
    cuda_devices = {v.device for v in tensors if v.device.type == "cuda"}

    def synchronize() -> None:
        for device in cuda_devices:
            torch.cuda.synchronize(device)

    start = time.perf_counter()
    synchronize()
    initial_sync_ms = (time.perf_counter() - start) * 1000
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    report: dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ipython": _version("ipython"),
        "transformer_lens": _version("transformer-lens"),
        "probe_repository_commit": commit.stdout.strip() or None,
        "cache_class": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "cache_class_file": inspect.getfile(type(cache)),
        "repr_source": inspect.getsource(type(cache).__repr__),
        "keys": len(cache.cache_dict),
        "tensor_entries": len(tensors),
        "logical_tensor_bytes": sum(v.numel() * v.element_size() for v in tensors),
        "devices": sorted({str(v.device) for v in tensors}),
        "print_options": {
            key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in vars(torch._tensor_str.PRINT_OPTS).items()
        },
        "initial_cuda_sync_ms": initial_sync_ms,
        "repeats": repeats,
        "measurements": {},
        "scope": "Formatting and captured publication only; no browser rendering. "
        "Logical bytes may double-count aliases. Cache construction is excluded.",
    }

    def printed() -> str:
        output = io.StringIO()
        with redirect_stdout(output):
            print(cache)
        return output.getvalue()

    cases: dict[str, Callable[[], Any]] = {
        "repr_wrapper": lambda: repr(cache),
        "str_wrapper": lambda: str(cache),
        "print_wrapper_to_memory": printed,
    }
    try:
        from IPython import get_ipython
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.display import display
        from IPython.utils.capture import capture_output

        shell = get_ipython() or InteractiveShell.instance()
        formatter = shell.display_formatter
        if formatter is None:
            raise RuntimeError("IPython has no display formatter")
        cases["ipython_format_wrapper"] = lambda: formatter.format(cache)

        def captured_display() -> None:
            with capture_output():
                display(cache)

        cases["display_wrapper_captured"] = captured_display
        if include_values:
            cases["ipython_format_dictionary"] = lambda: formatter.format(cache.cache_dict)
    except ImportError:
        report["ipython_status"] = "unavailable; display cases omitted"

    if include_values:
        cases["repr_dictionary"] = lambda: repr(cache.cache_dict)
        if tensors:
            cases["repr_first_tensor"] = lambda: repr(tensors[0])
            cases["repr_tensor_and_cache_tuple"] = lambda: repr((tensors[0], cache))

    for name, operation in cases.items():
        try:
            operation()  # Warm up separately from samples.
            durations = []
            result = None
            for _ in range(repeats):
                synchronize()
                start = time.perf_counter()
                result = operation()
                synchronize()
                durations.append((time.perf_counter() - start) * 1000)
            report["measurements"][name] = {
                "median_ms": statistics.median(durations),
                "samples_ms": durations,
                "text_chars": len(result) if isinstance(result, str) else None,
            }
        except Exception as error:
            report["measurements"][name] = {"error": f"{type(error).__name__}: {error}"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-values", action="store_true")
    parser.add_argument("--profile", choices=["default", "short", "full"], default="default")
    args = parser.parse_args()
    if args.keys < 0 or args.elements < 0 or args.repeats < 1:
        parser.error("keys/elements must be nonnegative and repeats must be positive")

    from transformer_lens import ActivationCache

    original_options = vars(torch._tensor_str.PRINT_OPTS).copy()
    try:
        torch.set_printoptions(profile=args.profile)
        cache = ActivationCache(
            {
                f"blocks.{i}.hook_resid_pre": torch.ones(args.elements, device=args.device)
                for i in range(args.keys)
            },
            model=None,
        )
        print(json.dumps(probe_cache(cache, args.repeats, args.include_values), indent=2))
    finally:
        torch.set_printoptions(**original_options)


if __name__ == "__main__":
    main()
