"""Wall-clock timing helpers. Data generation and evaluation stay outside."""
import time
import tracemalloc
import numpy as np


def benchmark_functions(functions, *, repeats=7, warmups=2):
    for _ in range(warmups):
        for fn in functions.values():
            result = fn()
            del result
    values = {name: [] for name in functions}
    raw = []
    names = list(functions)
    for repeat in range(repeats):
        start_index = repeat % len(names)
        for name in names[start_index:] + names[:start_index]:
            start = time.perf_counter_ns()
            result = functions[name]()
            elapsed = (time.perf_counter_ns()-start)/1e6
            status = result.diagnostics.get("status", "ok")
            raw.append({"method": name, "repeat": repeat, "ms": elapsed,
                        "status": status, "iterations": result.diagnostics.get("iterations")})
            values[name].append(elapsed)
            del result
    summary = {name: {"median_ms": float(np.median(ms)),
                      "q25_ms": float(np.quantile(ms, .25)),
                      "q75_ms": float(np.quantile(ms, .75))} for name, ms in values.items()}
    return summary, raw


def evaluate_with_memory(fn, enabled=False):
    """One untimed run; traced allocations, not whole-process RSS."""
    if not enabled:
        return fn(), None
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / (1024**2)
