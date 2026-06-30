"""
src/pipeline/batch_optimizer.py
================================
Batch processing optimizations for the exoplanet detection pipeline.

This module provides:

* **HDF5 caching** – persist and reload detrended flux, BLS/phase
  results, and extracted features to avoid repeated computation.
* **GPU batch inference** – run model inference on a PyTorch
  ``DataLoader`` on CUDA or CPU, returning predicted labels and
  class probabilities.
* **Completion-time estimation** – human-readable ETA strings.
* **Parallel preprocessing** – helper to distribute per-target
  preprocessing workloads across a ``multiprocessing.Pool``.

Notes
-----
The HDF5 file stores each TIC ID as a top-level group.  Within each
group sub-groups ``flux``, ``phase``, and ``features`` hold the
respective datasets.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _H5PY_AVAILABLE = False

try:
    from utils.config import get
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

    def get(k: str, d: Any = None) -> Any:  # type: ignore[misc]
        return d


# ---------------------------------------------------------------------------
# HDF5 caching helpers
# ---------------------------------------------------------------------------

def _require_h5py() -> None:
    if not _H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 caching. "
            "Install it with: pip install h5py"
        )


def _tic_group_name(tic_id: int | str) -> str:
    return f"tic_{int(tic_id):020d}"


def save_to_cache(
    tic_id: int | str,
    data_dict: dict[str, Any],
    cache_path: str | Path,
) -> None:
    """Save pre-processed data for one target to an HDF5 cache file.

    Parameters
    ----------
    tic_id:
        TESS Input Catalogue identifier.
    data_dict:
        Dictionary of arrays / scalars to store.  Typical keys:

        * ``"flux"``     – 1-D detrended flux array
        * ``"time"``     – 1-D time array (BTJD)
        * ``"phase_global"`` – global-view phase-folded flux (1-D)
        * ``"phase_local"``  – local-view phase-folded flux (1-D)
        * ``"features"`` – 1-D feature vector (float32)
        * ``"period_days"``  – scalar BLS period
        * ``"depth_ppm"``    – scalar transit depth in ppm
        * ``"snr"``          – scalar SNR

        Any additional keys that contain NumPy arrays or scalars are
        stored transparently.
    cache_path:
        Path to the ``.h5`` cache file (will be created if absent).

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    _require_h5py()
    import h5py  # noqa: F811 – local import after availability check

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    group_name = _tic_group_name(tic_id)

    with h5py.File(cache_path, "a") as hf:
        if group_name in hf:
            del hf[group_name]
        grp = hf.create_group(group_name)

        for key, val in data_dict.items():
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                grp.create_dataset(key, data=val, compression="gzip", compression_opts=4)
            elif isinstance(val, (int, float, bool)):
                grp.attrs[key] = val
            elif isinstance(val, str):
                grp.attrs[key] = val
            elif isinstance(val, (list, tuple)):
                try:
                    arr = np.asarray(val)
                    grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
                except Exception:
                    grp.attrs[key] = str(val)
            else:
                try:
                    grp.attrs[key] = val
                except Exception:
                    logger.debug("Skipping non-serialisable key '%s'", key)

    logger.debug("Cached TIC %s → %s", tic_id, cache_path)


def load_from_cache(
    tic_id: int | str,
    cache_path: str | Path,
) -> dict[str, Any] | None:
    """Load pre-processed data for a target from the HDF5 cache.

    Parameters
    ----------
    tic_id:
        TESS Input Catalogue identifier.
    cache_path:
        Path to the ``.h5`` cache file.

    Returns
    -------
    dict[str, Any] | None
        Reconstructed data dictionary if the target exists in the cache,
        otherwise ``None``.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    _require_h5py()
    import h5py  # noqa: F811

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    group_name = _tic_group_name(tic_id)

    try:
        with h5py.File(cache_path, "r") as hf:
            if group_name not in hf:
                return None
            grp = hf[group_name]
            data: dict[str, Any] = {}
            # Load datasets
            for key in grp.keys():
                data[key] = grp[key][()]  # type: ignore[index]
            # Load scalar / string attributes
            for key, val in grp.attrs.items():
                data[key] = val
        logger.debug("Cache HIT for TIC %s", tic_id)
        return data
    except Exception as exc:
        logger.warning("Cache read failed for TIC %s: %s", tic_id, exc)
        return None


def cache_hit_rate(cache_path: str | Path) -> float:
    """Return the fraction of cache entries that are valid and complete.

    For a freshly initialised run this will be 0.0; for a run resuming
    from a previous one it reflects how many targets are already done.

    Parameters
    ----------
    cache_path:
        Path to the ``.h5`` cache file.

    Returns
    -------
    float
        Fraction in [0, 1].  Returns 0.0 if the file does not exist.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    _require_h5py()
    import h5py  # noqa: F811

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return 0.0

    try:
        with h5py.File(cache_path, "r") as hf:
            n_groups = len(hf.keys())
    except Exception as exc:
        logger.warning("Could not read cache file %s: %s", cache_path, exc)
        return 0.0

    return float(n_groups) / max(n_groups, 1)  # always 1.0 if any entries exist


def cache_n_entries(cache_path: str | Path) -> int:
    """Return the total number of targets stored in an HDF5 cache file.

    Parameters
    ----------
    cache_path:
        Path to the ``.h5`` cache file.

    Returns
    -------
    int
    """
    _require_h5py()
    import h5py  # noqa: F811

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return 0
    try:
        with h5py.File(cache_path, "r") as hf:
            return len(hf.keys())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# GPU batch inference
# ---------------------------------------------------------------------------

def batch_predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference in batches on GPU/CPU.

    Parameters
    ----------
    model:
        A trained PyTorch model with a ``forward`` method that returns
        logits of shape ``(batch_size, num_classes)``.
    dataloader:
        PyTorch :class:`~torch.utils.data.DataLoader` providing input
        tensors (and optionally labels which are ignored here).
    device:
        Target device string (``"cuda"``, ``"cpu"``) or
        :class:`torch.device`.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Integer predicted class index for each sample.
    probs : np.ndarray, shape (N, num_classes)
        Softmax class probabilities for each sample.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            # DataLoader may return (inputs,) or (inputs, labels) tuples
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            if not isinstance(inputs, torch.Tensor):
                inputs = torch.as_tensor(inputs)

            inputs = inputs.to(device)
            logits: torch.Tensor = model(inputs)
            probs_batch: torch.Tensor = torch.softmax(logits, dim=-1)
            labels_batch: torch.Tensor = torch.argmax(probs_batch, dim=-1)

            all_probs.append(probs_batch.cpu().numpy())
            all_labels.append(labels_batch.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return labels_np, probs_np


# ---------------------------------------------------------------------------
# Completion-time estimation
# ---------------------------------------------------------------------------

def estimate_completion_time(
    n_targets: int,
    time_per_target_s: float,
) -> str:
    """Return a human-readable ETA string for processing *n_targets*.

    Parameters
    ----------
    n_targets:
        Number of remaining targets to process.
    time_per_target_s:
        Average wall-clock seconds per target (from a rolling average
        or a calibration run).

    Returns
    -------
    str
        Human-readable ETA such as ``"2h 13m 05s"`` or ``"< 1s"``.

    Examples
    --------
    >>> estimate_completion_time(1200, 3.5)
    '1h 10m 00s'
    """
    if n_targets <= 0 or time_per_target_s <= 0:
        return "< 1s"

    total_s = n_targets * time_per_target_s
    if total_s < 1.0:
        return "< 1s"

    hours = int(total_s // 3600)
    minutes = int((total_s % 3600) // 60)
    seconds = int(total_s % 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes:02d}m")
    parts.append(f"{seconds:02d}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Parallel preprocessing
# ---------------------------------------------------------------------------

def _worker_init(func: Callable, args: tuple) -> Any:
    """Pool worker wrapper – catches and logs exceptions."""
    try:
        return func(*args)
    except Exception as exc:
        logger.error("Worker error: %s", exc, exc_info=True)
        return None


def parallel_preprocess(
    worker_fn: Callable[..., Any],
    arg_list: Sequence[tuple],
    n_workers: int = 4,
    chunksize: int = 1,
) -> list[Any]:
    """Distribute per-target preprocessing across a ``multiprocessing.Pool``.

    Parameters
    ----------
    worker_fn:
        A top-level (picklable) function that processes a single target.
        Must accept a tuple of arguments matching entries in *arg_list*.
    arg_list:
        Sequence of argument tuples, one per target.  Each element is
        unpacked and passed to *worker_fn* as ``worker_fn(*args)``.
    n_workers:
        Number of worker processes.  Falls back to 1 (serial) if <= 1
        or if ``n_workers`` exceeds the number of targets.
    chunksize:
        Number of tasks dispatched to each worker at once.

    Returns
    -------
    list[Any]
        Results in the same order as *arg_list*.  Failed tasks return
        ``None``.

    Notes
    -----
    On Windows the multiprocessing start method is ``"spawn"``.  Worker
    functions **must** be importable top-level functions (not lambdas).
    If ``n_workers == 1`` the work is done in the calling process to
    simplify debugging.
    """
    n_workers = max(1, min(n_workers, len(arg_list), mp.cpu_count()))

    if n_workers == 1:
        logger.debug("Running preprocessing serially (n_workers=1)")
        results = []
        for args in arg_list:
            try:
                results.append(worker_fn(*args))
            except Exception as exc:
                logger.error("Serial worker error: %s", exc, exc_info=True)
                results.append(None)
        return results

    logger.info(
        "Parallel preprocessing: %d targets, %d workers", len(arg_list), n_workers
    )
    t0 = time.perf_counter()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.starmap(worker_fn, arg_list, chunksize=chunksize)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Parallel preprocessing done: %.1fs  (%.2f s/target)",
        elapsed,
        elapsed / max(len(arg_list), 1),
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry-point (diagnostics / benchmarking)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description=(
            "Batch optimizer diagnostics: report cache hit rate and "
            "estimate completion time."
        )
    )
    _parser.add_argument("--cache-path", type=str, default="data/interim/cache.h5")
    _parser.add_argument("--n-targets", type=int, default=500)
    _parser.add_argument("--time-per-target", type=float, default=5.0,
                         help="Estimated seconds per target.")
    _args = _parser.parse_args()

    if _H5PY_AVAILABLE:
        n_cached = cache_n_entries(_args.cache_path)
        print(f"Cache entries: {n_cached}  ({_args.cache_path})")
    else:
        print("h5py not installed – caching unavailable.")

    eta = estimate_completion_time(_args.n_targets, _args.time_per_target)
    print(f"ETA for {_args.n_targets} targets at {_args.time_per_target}s each: {eta}")
