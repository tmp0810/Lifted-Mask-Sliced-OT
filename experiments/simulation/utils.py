"""Simulation configuration, CSV I/O and provenance; no transport solvers."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess

from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[2]


def save_table(path, rows, delimiter=","):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def metadata():
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "PyYAML", "threadpoolctl", "POT", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            return None

    files = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "experiments").rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "packages": packages,
            "threadpools": threadpool_info(), "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"), "source_sha256": digest.hexdigest(),
            "cost": "squared Euclidean, excluding entropy; bounding-box cost <= 1"}


def validate_config(cfg):
    if cfg["scenario"] not in ("support", "weights"):
        raise ValueError("unknown scenario")
    values = cfg["overlap_values" if cfg["scenario"] == "support" else "weight_change_values"]
    if not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError("sweep values must be in [0,1]")
    for name in ("sizes", "dimensions", "projection_counts", "seeds", "projection_kinds", "weight_modes"):
        if not cfg[name] or len(set(cfg[name])) != len(cfg[name]):
            raise ValueError(f"{name} must be nonempty and contain no duplicates")
    if min(cfg["sizes"]) < 2 or min(cfg["dimensions"]) < 2 or min(cfg["projection_counts"]) < 1:
        raise ValueError("invalid simulation size/dimension/projection count")
    if cfg["repeats"] < 1 or cfg["warmups"] < 0 or cfg["threads"] < 1:
        raise ValueError("invalid timing settings")
    if cfg["sinkhorn"]["backend"] not in ("pot", "torch_log"):
        raise ValueError("Sinkhorn backend must be pot or torch_log")
    if cfg["sinkhorn"]["backend"] == "pot":
        import ot  # Fail early if the configured backend is not installed.
        if cfg["sinkhorn"].get("max_seconds") is not None:
            raise ValueError("POT requires max_seconds: null; use max_iter as its budget")
    return values
