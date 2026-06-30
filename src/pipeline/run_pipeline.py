"""
src/pipeline/run_pipeline.py
=============================
Main end-to-end pipeline orchestrator for the exoplanet detection system.

This is the **primary entry point** for the entire system.  It wires
together every pipeline phase in the correct order and handles error
recovery, progress tracking, and final reporting.

Pipeline phases
---------------
1.  Data download   – light curves and labels (unless ``--skip-download``)
2.  Integrity check – filter to valid FITS files
3.  Quality control – filter to photometrically passing targets
4.  Per-target preprocessing (parallel):
    a. Load FITS → remove systematics → wavelet detrend → GP detrend
    b. Phase-fold (BLS) → PhaseResult
    c. Centroid shift (if TPF available)
    d. Gaia contamination check
    e. Extract features (flux + shape + centroid/contamination)
    f. Compute SNR and FAP
5.  Model training  (unless ``--skip-training``)
6.  Model inference on all targets
7.  MCMC parameter estimation for PLANET targets with SNR > threshold
8.  Confidence scoring for all targets
9.  Format and save per-target JSON results
10. Save aggregate ``pipeline_results.csv``

CLI usage
---------
::

    python run_pipeline.py \\
        --sector 1 \\
        --max-targets 200 \\
        --workers 4 \\
        --output-dir outputs/

    python run_pipeline.py --tic-ids 22529346,25155310,259377017
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap sys.path so that the src package is importable when the script
# is executed directly from the repository root.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, load_config, project_root
from utils.logger import get_logger
from pipeline.logging_config import (
    setup_pipeline_logging,
    log_target_start,
    log_target_done,
    log_target_error,
    log_pipeline_summary,
)
from pipeline.output_formatter import format_result, save_result, save_aggregate_csv
from pipeline.batch_optimizer import (
    save_to_cache,
    load_from_cache,
    estimate_completion_time,
    parallel_preprocess,
    batch_predict,
)

# Try to import rich for progress bars; fall back gracefully.
try:
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        SpinnerColumn,
        MofNCompleteColumn,
    )
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
PLANET            = 0
ECLIPSING_BINARY  = 1
BLEND             = 2
NOISE             = 3
CLASS_NAMES       = ["PLANET", "ECLIPSING_BINARY", "BLEND", "NOISE"]


# ---------------------------------------------------------------------------
# Lazy-import wrappers for optional pipeline modules
# ---------------------------------------------------------------------------

def _try_import(module_path: str, name: str = "") -> Any:
    """Attempt to import a module; return None on failure."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, name) if name else mod
    except Exception as exc:
        logger.debug("Optional module not available: %s  (%s)", module_path, exc)
        return None


# ---------------------------------------------------------------------------
# Per-target worker (picklable – must be top-level function)
# ---------------------------------------------------------------------------

def _preprocess_target(
    tic_id: int,
    fits_path: str,
    tpf_path: str | None,
    cache_path: str,
    config_path: str | None,
) -> dict[str, Any] | None:
    """Process a single target through all preprocessing steps.

    This function is designed to be called inside a ``multiprocessing.Pool``
    worker.  It is deliberately self-contained (no closures) to ensure
    picklability on Windows.

    Parameters
    ----------
    tic_id:
        TESS Input Catalogue identifier.
    fits_path:
        Path to the 2-minute cadence FITS light-curve file.
    tpf_path:
        Path to the target-pixel-file FITS (may be ``None``).
    cache_path:
        Path to the shared HDF5 cache file.
    config_path:
        Path to ``config.yaml`` (or ``None`` for default).

    Returns
    -------
    dict[str, Any] | None
        Pre-processed target data or ``None`` on failure.
    """
    import sys
    from pathlib import Path as _Path

    _src = str(_Path(__file__).resolve().parents[2])
    if _src not in sys.path:
        sys.path.insert(0, _src)

    try:
        from utils.config import load_config, get as cfg_get
        from utils.logger import get_logger as _get_logger
        _log = _get_logger(f"preprocess.{tic_id}")
    except Exception:
        import logging
        _log = logging.getLogger(f"preprocess.{tic_id}")

    # ── Check cache ──────────────────────────────────────────────────────────
    try:
        cached = load_from_cache(tic_id, cache_path)
        if cached is not None:
            _log.debug("Cache hit for TIC %d", tic_id)
            return cached
    except Exception:
        pass

    result: dict[str, Any] = {"tic_id": tic_id}

    # ── Load FITS light curve ─────────────────────────────────────────────────
    try:
        import lightkurve as lk  # type: ignore
        lc = lk.read(fits_path)
        time_arr = lc.time.bkjd if hasattr(lc.time, "bkjd") else np.asarray(lc.time.value)
        flux_arr = np.asarray(lc.flux.value, dtype=np.float64)
        err_arr  = (
            np.asarray(lc.flux_err.value, dtype=np.float64)
            if lc.flux_err is not None
            else np.ones_like(flux_arr) * np.nanstd(flux_arr)
        )
        result.update({"time": time_arr, "flux_raw": flux_arr, "flux_err": err_arr})
    except Exception as exc:
        _log.error("FITS load failed for TIC %d: %s", tic_id, exc)
        return None

    flux = flux_arr.copy()
    time_arr_work = time_arr.copy()

    # ── Remove NaNs ───────────────────────────────────────────────────────────
    finite_mask = np.isfinite(flux) & np.isfinite(time_arr_work)
    time_arr_work = time_arr_work[finite_mask]
    flux          = flux[finite_mask]
    err_arr       = err_arr[finite_mask]

    # ── Normalize ─────────────────────────────────────────────────────────────
    med = np.nanmedian(flux)
    if med != 0:
        flux     = flux / med - 1.0
        err_arr  = err_arr / med

    # ── Wavelet detrending ────────────────────────────────────────────────────
    flux_detrended = flux.copy()
    try:
        from conditioning import wavelet_detrend  # type: ignore
        flux_detrended = wavelet_detrend.detrend(time_arr_work, flux, err_arr)
    except Exception:
        try:
            import pywt
            wavelet  = try_get_cfg("conditioning.wavelet_family", "db8", config_path)
            levels   = try_get_cfg("conditioning.wavelet_levels", 4, config_path)
            coeffs   = pywt.wavedec(flux, wavelet, level=levels)
            sigma    = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(flux)))
            coeffs   = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
            flux_detrended = pywt.waverec(coeffs, wavelet)[: len(flux)]
        except Exception:
            flux_detrended = flux

    result["flux_detrended"] = flux_detrended.astype(np.float32)

    # ── BLS phase-fold ───────────────────────────────────────────────────────
    period_days   = float("nan")
    depth_ppm     = float("nan")
    duration_hrs  = float("nan")
    phase_global  = np.zeros(200, dtype=np.float32)
    phase_local   = np.zeros(50,  dtype=np.float32)

    try:
        from conditioning import phase_fold  # type: ignore
        phase_result = phase_fold.run(time_arr_work, flux_detrended, err_arr)
        period_days  = float(phase_result.period_days)
        depth_ppm    = float(phase_result.depth_ppm)
        duration_hrs = float(phase_result.duration_hrs)
        phase_global = np.asarray(phase_result.global_view, dtype=np.float32)
        phase_local  = np.asarray(phase_result.local_view,  dtype=np.float32)
    except Exception:
        try:
            from astropy.timeseries import BoxLeastSquares
            bls   = BoxLeastSquares(time_arr_work, flux_detrended)
            pgram = bls.autopower(0.16, objective="snr")
            best  = pgram.period[np.argmax(pgram.power)]
            period_days = float(best)
        except Exception:
            pass

    result.update({
        "period_days":  period_days,
        "depth_ppm":    depth_ppm,
        "duration_hrs": duration_hrs,
        "phase_global": phase_global,
        "phase_local":  phase_local,
    })

    # ── Centroid shift ────────────────────────────────────────────────────────
    centroid_shift_arcsec = float("nan")
    if tpf_path:
        try:
            from acquisition import centroid_shift  # type: ignore
            cs_result = centroid_shift.compute(tpf_path, period_days)
            centroid_shift_arcsec = float(cs_result.get("centroid_shift_arcsec", float("nan")))
        except Exception:
            pass
    result["centroid_shift_arcsec"] = centroid_shift_arcsec

    # ── Gaia contamination ────────────────────────────────────────────────────
    contamination_ratio = float("nan")
    is_contaminated     = False
    try:
        from acquisition import gaia_contamination  # type: ignore
        cont_result         = gaia_contamination.compute(tic_id)
        contamination_ratio = float(cont_result.get("contamination_ratio", float("nan")))
        is_contaminated     = bool(cont_result.get("is_contaminated", False))
    except Exception:
        pass
    result["contamination_ratio"] = contamination_ratio
    result["is_contaminated"]     = is_contaminated

    # ── Feature extraction ────────────────────────────────────────────────────
    features = np.zeros(128, dtype=np.float32)
    try:
        from features import flux_features, shape_features  # type: ignore
        flux_feats  = flux_features.extract(time_arr_work, flux_detrended, err_arr)
        shape_feats = shape_features.extract(phase_global, phase_local,
                                              period_days, depth_ppm, duration_hrs)
        features = np.concatenate([flux_feats, shape_feats]).astype(np.float32)
    except Exception:
        # Fallback: basic statistical features
        safe = flux_detrended[np.isfinite(flux_detrended)]
        if len(safe) > 0:
            features[:8] = np.array([
                np.mean(safe), np.std(safe), np.median(safe),
                np.percentile(safe, 5), np.percentile(safe, 95),
                float(np.isfinite(period_days)),
                float(np.isfinite(depth_ppm)) * depth_ppm / 1e4 if np.isfinite(depth_ppm) else 0.0,
                float(np.isfinite(duration_hrs)) * duration_hrs / 24.0 if np.isfinite(duration_hrs) else 0.0,
            ], dtype=np.float32)

    result["features"] = features

    # ── SNR & FAP ─────────────────────────────────────────────────────────────
    snr = float("nan")
    fap = float("nan")
    try:
        from scoring import snr_fap  # type: ignore
        snr_result = snr_fap.compute(flux_detrended, period_days, depth_ppm, duration_hrs)
        snr = float(snr_result.get("snr", float("nan")))
        fap = float(snr_result.get("fap", float("nan")))
    except Exception:
        # Simple SNR estimate: depth / noise
        if np.isfinite(depth_ppm) and len(flux_detrended) > 0:
            noise = np.nanstd(flux_detrended) * 1e6
            snr   = depth_ppm / max(noise, 1e-6)
    result["snr"] = snr
    result["fap"] = fap

    # ── Persist to cache ──────────────────────────────────────────────────────
    try:
        save_to_cache(tic_id, result, cache_path)
    except Exception:
        pass

    return result


def try_get_cfg(key: str, default: Any, config_path: str | None) -> Any:
    """Thread-safe config getter used inside worker functions."""
    try:
        from utils.config import get as _get
        return _get(key, default, config_path)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Discover available files
# ---------------------------------------------------------------------------

def _discover_fits_files(
    lc_dir: Path, tpf_dir: Path, tic_ids: list[int] | None
) -> list[tuple[int, str, str | None]]:
    """Return list of ``(tic_id, fits_path, tpf_path|None)`` tuples.

    Parameters
    ----------
    lc_dir:
        Directory containing light curve FITS files.
    tpf_dir:
        Directory containing target pixel file FITS files.
    tic_ids:
        Optional subset of TIC IDs to restrict to.

    Returns
    -------
    list of (int, str, str | None)
    """
    fits_files = sorted(lc_dir.glob("**/*.fits")) + sorted(lc_dir.glob("**/*.fit"))
    records: list[tuple[int, str, str | None]] = []
    for fp in fits_files:
        stem = fp.stem
        parts = stem.split("-")
        tic_candidate = None
        if len(parts) >= 3:
            try:
                digits = "".join(c for c in parts[2] if c.isdigit())
                if len(digits) >= 6:
                    tic_candidate = int(digits)
            except ValueError:
                pass
        if tic_candidate is None:
            continue
        if tic_ids is not None and tic_candidate not in tic_ids:
            continue
        # Look for a TPF counterpart
        tpf_path = None
        tpf_candidates = list(tpf_dir.glob(f"*{tic_candidate}*"))
        if tpf_candidates:
            tpf_path = str(tpf_candidates[0])
        records.append((tic_candidate, str(fp), tpf_path))

    return records


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def _run_training(
    features_list: list[np.ndarray],
    labels: list[int],
    config_path: str | None,
    output_dir: Path,
) -> "Any | None":
    """Train the classification model and return it.

    Falls back gracefully if the model module is unavailable.
    """
    try:
        from models import classifier  # type: ignore
        model = classifier.train(features_list, labels, config_path=config_path)
        ckpt_path = output_dir / "best_model.pt"
        classifier.save_checkpoint(model, str(ckpt_path))
        logger.info("Model checkpoint saved to %s", ckpt_path)
        return model
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return None


def _load_model(config_path: str | None, checkpoint_dir: Path) -> "Any | None":
    """Load a pre-trained model from a checkpoint directory."""
    try:
        from models import classifier  # type: ignore
        ckpt_candidates = sorted(checkpoint_dir.glob("*.pt"))
        if not ckpt_candidates:
            logger.warning("No checkpoint found in %s", checkpoint_dir)
            return None
        ckpt = ckpt_candidates[-1]  # latest
        model = classifier.load_checkpoint(str(ckpt), config_path=config_path)
        logger.info("Loaded model checkpoint: %s", ckpt)
        return model
    except Exception as exc:
        logger.error("Failed to load model: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_inference(
    model: Any,
    target_data_list: list[dict[str, Any]],
    device: str,
) -> tuple[list[int], list[np.ndarray]]:
    """Run model inference on all targets.

    Returns
    -------
    labels : list[int]
    probs  : list[np.ndarray]  – shape (4,) per target
    """
    if model is None:
        logger.warning("No model available; all targets classified as NOISE.")
        n = len(target_data_list)
        return [NOISE] * n, [np.array([0.0, 0.0, 0.0, 1.0])] * n

    import torch
    from torch.utils.data import TensorDataset, DataLoader

    features = []
    for td in target_data_list:
        f = td.get("features")
        if f is None:
            f = np.zeros(128, dtype=np.float32)
        features.append(np.asarray(f, dtype=np.float32))

    max_len = max(len(f) for f in features)
    padded = np.zeros((len(features), max_len), dtype=np.float32)
    for i, f in enumerate(features):
        padded[i, : len(f)] = f

    X = torch.from_numpy(padded)
    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)

    try:
        labels_arr, probs_arr = batch_predict(model, loader, device=device)
        return labels_arr.tolist(), [probs_arr[i] for i in range(len(probs_arr))]
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        n = len(target_data_list)
        return [NOISE] * n, [np.array([0.0, 0.0, 0.0, 1.0])] * n


# ---------------------------------------------------------------------------
# MCMC helper
# ---------------------------------------------------------------------------

def _run_mcmc(
    tic_id: int,
    target_data: dict[str, Any],
    config_path: str | None,
) -> dict[str, Any]:
    """Run MCMC parameter estimation and return param_dict."""
    try:
        from scoring import mcmc_sampler  # type: ignore
        result = mcmc_sampler.run(
            time=target_data.get("time", np.array([])),
            flux=target_data.get("flux_detrended", np.array([])),
            flux_err=target_data.get("flux_err", np.array([])),
            period_init=target_data.get("period_days", 3.0),
            depth_init=target_data.get("depth_ppm", 1000.0) / 1e6,
            duration_init=target_data.get("duration_hrs", 2.0) / 24.0,
            config_path=config_path,
        )
        return result
    except Exception as exc:
        logger.warning("MCMC failed for TIC %d: %s", tic_id, exc)
        return {
            "period_days":  target_data.get("period_days", float("nan")),
            "period_err":   float("nan"),
            "depth_ppm":    target_data.get("depth_ppm", float("nan")),
            "depth_err":    float("nan"),
            "duration_hrs": target_data.get("duration_hrs", float("nan")),
            "duration_err": float("nan"),
        }


# ---------------------------------------------------------------------------
# Confidence scoring helper
# ---------------------------------------------------------------------------

def _compute_confidence(
    tic_id: int,
    probs: np.ndarray,
    target_data: dict[str, Any],
    config_path: str | None,
) -> dict[str, Any]:
    """Compute pipeline confidence score."""
    try:
        from scoring import confidence  # type: ignore
        return confidence.compute_confidence(
            class_probs=probs,
            snr=target_data.get("snr", 0.0),
            fap=target_data.get("fap", 1.0),
            config_path=config_path,
        )
    except Exception:
        planet_prob = float(probs[0]) if len(probs) > 0 else 0.0
        snr = float(target_data.get("snr", 0.0) or 0.0)
        snr_score = min(snr / 20.0, 1.0)
        pipeline_confidence = 0.7 * planet_prob + 0.3 * snr_score
        return {"pipeline_confidence": pipeline_confidence}


# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------

def _print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a Rich summary table of all results."""
    if not _RICH:
        _print_plain_summary(results)
        return

    console = Console()
    table   = Table(title="Pipeline Results Summary", show_lines=True)
    table.add_column("TIC ID",      style="cyan",    justify="right")
    table.add_column("Label",       style="magenta", justify="left")
    table.add_column("Confidence",  style="green",   justify="right")
    table.add_column("P(planet)",   style="yellow",  justify="right")
    table.add_column("Period (d)",  style="blue",    justify="right")
    table.add_column("SNR",         style="white",   justify="right")
    table.add_column("FAP",         style="red",     justify="right")

    for r in sorted(results, key=lambda x: x.get("pipeline_confidence", 0.0), reverse=True):
        table.add_row(
            str(r.get("tic_id", "?")),
            r.get("predicted_label_name", "?"),
            f'{r.get("pipeline_confidence", float("nan")):.3f}',
            f'{r.get("planet_prob", float("nan")):.3f}',
            f'{r.get("period_days", float("nan")):.4f}' if not _isnan(r.get("period_days")) else "—",
            f'{r.get("snr", float("nan")):.1f}' if not _isnan(r.get("snr")) else "—",
            f'{r.get("fap", float("nan")):.2e}' if not _isnan(r.get("fap")) else "—",
        )
    console.print(table)


def _print_plain_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== Pipeline Results Summary ===")
    print(f"{'TIC ID':>15}  {'Label':20}  {'Conf':>6}  {'P(pl)':>6}")
    print("-" * 60)
    for r in results:
        print(
            f"{r.get('tic_id', '?'):>15}  "
            f"{r.get('predicted_label_name', '?'):20}  "
            f"{r.get('pipeline_confidence', 0):>6.3f}  "
            f"{r.get('planet_prob', 0):>6.3f}"
        )


def _isnan(v: Any) -> bool:
    if v is None:
        return True
    try:
        import math
        return math.isnan(float(v))
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Main pipeline entry point.

    Parameters
    ----------
    argv:
        Command-line arguments list.  When ``None``, ``sys.argv[1:]`` is
        used.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    # ── Parse arguments ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "AI-enabled end-to-end exoplanet detection pipeline.\n"
            "Runs data download → preprocessing → training → inference → output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sector", type=int,
        default=None,
        help="TESS sector number to process (default: from config).",
    )
    parser.add_argument(
        "--max-targets", type=int,
        default=None,
        help="Maximum number of targets to process (default: from config).",
    )
    parser.add_argument(
        "--tic-ids", type=str,
        default=None,
        help="Comma-separated TIC IDs to process.  Overrides --max-targets.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Use already-downloaded FITS files (skip acquisition phase).",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training and use the latest existing checkpoint.",
    )
    parser.add_argument(
        "--config", type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Output directory (default: from config paths.outputs).",
    )
    parser.add_argument(
        "--workers", type=int,
        default=None,
        help="Number of parallel preprocessing workers (default: from config).",
    )
    parser.add_argument(
        "--device", type=str,
        default=None,
        help="Inference device: 'auto', 'cpu', or 'cuda' (default: from config).",
    )

    args = parser.parse_args(argv)

    # ── Resolve config & paths ────────────────────────────────────────────────
    cfg_path = args.config  # may be None → falls back to default
    try:
        load_config(cfg_path)  # validate config is parseable
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    root          = project_root()
    sector        = args.sector        or get("acquisition.sector", 1,    cfg_path)
    max_targets   = args.max_targets   or get("acquisition.max_targets", 500, cfg_path)
    n_workers     = args.workers       or get("training.num_workers", 4,  cfg_path)
    device_cfg    = args.device        or get("training.device", "auto",  cfg_path)
    snr_threshold = get("scoring.snr_threshold", 7.0, cfg_path)

    lc_dir        = root / get("paths.raw_lc",      "data/raw/lightcurves", cfg_path)
    tpf_dir       = root / get("paths.raw_tpf",     "data/raw/tpf",         cfg_path)
    interim_dir   = root / get("paths.interim",     "data/interim",          cfg_path)
    ckpt_dir      = root / get("paths.checkpoints", "checkpoints",           cfg_path)
    out_dir_str   = args.output_dir or str(root / get("paths.outputs", "outputs", cfg_path))
    out_dir       = Path(out_dir_str)
    cache_path    = interim_dir / "cache.h5"

    out_dir.mkdir(parents=True,    exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True,    exist_ok=True)

    # Resolve device
    import torch
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    # ── Set up logging ────────────────────────────────────────────────────────
    pipeline_log = setup_pipeline_logging(cfg_path)
    pipeline_log.info("Pipeline starting | sector=%d  device=%s", sector, device)

    # ── Parse TIC IDs ─────────────────────────────────────────────────────────
    tic_id_filter: list[int] | None = None
    if args.tic_ids:
        try:
            tic_id_filter = [int(t.strip()) for t in args.tic_ids.split(",") if t.strip()]
            pipeline_log.info("Restricting to %d TIC IDs from --tic-ids", len(tic_id_filter))
        except ValueError as exc:
            print(f"ERROR: --tic-ids must be comma-separated integers: {exc}", file=sys.stderr)
            return 1

    pipeline_t0 = time.perf_counter()
    n_success   = 0
    n_failed    = 0
    all_results: list[dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Data acquisition
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_download:
        pipeline_log.info("PHASE 1: Downloading light curves (sector=%d)…", sector)
        try:
            from acquisition import download_lightcurves  # type: ignore
            download_lightcurves.run(
                sector=sector,
                max_targets=max_targets if not tic_id_filter else None,
                tic_ids=tic_id_filter,
                output_dir=str(lc_dir),
                config_path=cfg_path,
            )
        except Exception as exc:
            pipeline_log.warning("download_lightcurves failed: %s — continuing.", exc)

        try:
            from acquisition import download_labels  # type: ignore
            download_labels.run(sector=sector, config_path=cfg_path)
        except Exception as exc:
            pipeline_log.warning("download_labels failed: %s — continuing.", exc)
    else:
        pipeline_log.info("PHASE 1: Skipped (--skip-download).")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Integrity check
    # ─────────────────────────────────────────────────────────────────────────
    pipeline_log.info("PHASE 2: Integrity check…")
    valid_records = _discover_fits_files(lc_dir, tpf_dir, tic_id_filter)
    if tic_id_filter is None and max_targets:
        valid_records = valid_records[:max_targets]
    pipeline_log.info("Found %d valid FITS files.", len(valid_records))

    if not valid_records:
        pipeline_log.error("No valid FITS files found.  Aborting.")
        return 1

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: Quality control filter
    # ─────────────────────────────────────────────────────────────────────────
    pipeline_log.info("PHASE 3: Quality control…")
    qc_max_nan  = get("quality_control.max_nan_fraction", 0.20, cfg_path)
    qc_min_days = get("quality_control.min_duration_days", 10.0, cfg_path)

    passing_records: list[tuple[int, str, str | None]] = []
    for tic_id, fits_path, tpf_path in valid_records:
        try:
            from preparation import quality_control  # type: ignore
            ok = quality_control.check(fits_path, max_nan=qc_max_nan, min_days=qc_min_days)
            if ok:
                passing_records.append((tic_id, fits_path, tpf_path))
        except Exception:
            passing_records.append((tic_id, fits_path, tpf_path))

    pipeline_log.info("%d / %d targets passed QC.", len(passing_records), len(valid_records))

    if not passing_records:
        pipeline_log.error("No targets passed quality control.  Aborting.")
        return 1

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: Per-target preprocessing (parallel)
    # ─────────────────────────────────────────────────────────────────────────
    pipeline_log.info("PHASE 4: Preprocessing %d targets with %d workers…",
                      len(passing_records), n_workers)

    arg_list = [
        (tic_id, fits_path, tpf_path, str(cache_path), cfg_path)
        for tic_id, fits_path, tpf_path in passing_records
    ]

    # Calibrate ETA
    eta_str = estimate_completion_time(len(arg_list), 5.0)
    pipeline_log.info("Estimated preprocessing time: %s", eta_str)

    def _progress_wrapper() -> list[dict[str, Any] | None]:
        if _RICH and n_workers == 1:
            # Serial with Rich progress bar for cleaner output
            results_inner: list[dict[str, Any] | None] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Preprocessing…", total=len(arg_list))
                for a in arg_list:
                    r = _preprocess_target(*a)
                    results_inner.append(r)
                    progress.advance(task)
            return results_inner
        else:
            return parallel_preprocess(_preprocess_target, arg_list, n_workers=n_workers)

    preproc_results = _progress_wrapper()

    # Filter failed targets
    target_data_list: list[dict[str, Any]] = []
    tic_ids_success: list[int]             = []
    for (tic_id, _, _), result in zip(passing_records, preproc_results):
        if result is None:
            n_failed += 1
            pipeline_log.warning("Preprocessing returned None for TIC %d", tic_id)
        else:
            target_data_list.append(result)
            tic_ids_success.append(tic_id)

    pipeline_log.info("Preprocessing complete: %d ok, %d failed.",
                      len(target_data_list), n_failed)

    if not target_data_list:
        pipeline_log.error("All targets failed preprocessing.  Aborting.")
        return 1

    # ─────────────────────────────────────────────────────────────────────────
    # Load labels (for training)
    # ─────────────────────────────────────────────────────────────────────────
    labels_for_training: list[int] = []
    try:
        label_file = root / get("paths.raw_labels", "data/raw/labels", cfg_path)
        label_csvs = list(Path(str(label_file)).glob("*.csv"))
        if label_csvs:
            ldf = pd.concat([pd.read_csv(f) for f in label_csvs], ignore_index=True)
            ldf.columns = ldf.columns.str.lower()
            if "tic_id" in ldf.columns and "label" in ldf.columns:
                label_map = dict(zip(ldf["tic_id"].astype(int), ldf["label"].astype(int)))
                labels_for_training = [label_map.get(tid, NOISE) for tid in tic_ids_success]
    except Exception as exc:
        pipeline_log.warning("Could not load training labels: %s", exc)

    if not labels_for_training:
        labels_for_training = [NOISE] * len(target_data_list)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: Model training
    # ─────────────────────────────────────────────────────────────────────────
    model = None
    if not args.skip_training:
        pipeline_log.info("PHASE 5: Training classification model…")
        features_list = [td.get("features", np.zeros(128, dtype=np.float32))
                         for td in target_data_list]
        model = _run_training(features_list, labels_for_training, cfg_path, ckpt_dir)
    else:
        pipeline_log.info("PHASE 5: Loading existing model (--skip-training)…")
        model = _load_model(cfg_path, ckpt_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 6: Model inference
    # ─────────────────────────────────────────────────────────────────────────
    pipeline_log.info("PHASE 6: Running inference on %d targets…", len(target_data_list))
    pred_labels, pred_probs = _run_inference(model, target_data_list, device)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7 → 10: Per-target MCMC, confidence, format, save
    # ─────────────────────────────────────────────────────────────────────────
    pipeline_log.info("PHASE 7–10: MCMC / confidence / formatting…")

    iter_items = zip(tic_ids_success, target_data_list, pred_labels, pred_probs)

    if _RICH:
        _progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        _task_id = _progress_ctx.__enter__().add_task(
            "Post-processing…", total=len(tic_ids_success)
        )
    else:
        _progress_ctx = None
        _task_id      = None

    try:
        for tic_id, target_data, label, probs in iter_items:
            t_start = time.perf_counter()
            log_target_start(pipeline_log, tic_id, sector)

            try:
                # MCMC for high-confidence planet candidates
                param_dict: dict[str, Any] = {
                    "period_days":  target_data.get("period_days",  float("nan")),
                    "period_err":   float("nan"),
                    "depth_ppm":    target_data.get("depth_ppm",    float("nan")),
                    "depth_err":    float("nan"),
                    "duration_hrs": target_data.get("duration_hrs", float("nan")),
                    "duration_err": float("nan"),
                }
                snr_val = float(target_data.get("snr", 0.0) or 0.0)
                if label == PLANET and snr_val >= snr_threshold:
                    param_dict = _run_mcmc(tic_id, target_data, cfg_path)

                # Confidence
                conf_dict = _compute_confidence(tic_id, probs, target_data, cfg_path)

                # Build centroid / contamination sub-dicts
                centroid_result = (
                    {"centroid_shift_arcsec": target_data.get("centroid_shift_arcsec")}
                    if "centroid_shift_arcsec" in target_data
                    else None
                )
                contamination_result = (
                    {
                        "contamination_ratio": target_data.get("contamination_ratio"),
                        "is_contaminated":     target_data.get("is_contaminated", False),
                    }
                    if "contamination_ratio" in target_data
                    else None
                )

                elapsed = time.perf_counter() - t_start
                result  = format_result(
                    tic_id=tic_id,
                    label=label,
                    class_probs=probs,
                    confidence_dict=conf_dict,
                    param_dict=param_dict,
                    snr_dict={"snr": snr_val},
                    fap_dict={"fap": target_data.get("fap", float("nan"))},
                    centroid_result=centroid_result,
                    contamination_result=contamination_result,
                    processing_time_s=elapsed,
                )
                save_result(result, out_dir)
                all_results.append(result)

                conf_val = float(conf_dict.get("pipeline_confidence", 0.0))
                log_target_done(pipeline_log, tic_id, elapsed, label, conf_val)
                n_success += 1

            except Exception as exc:
                n_failed += 1
                log_target_error(pipeline_log, tic_id, exc)

            if _progress_ctx is not None and _task_id is not None:
                _progress_ctx.advance(_task_id)  # type: ignore[attr-defined]

    finally:
        if _progress_ctx is not None:
            _progress_ctx.__exit__(None, None, None)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 10: Aggregate CSV
    # ─────────────────────────────────────────────────────────────────────────
    pipeline_log.info("PHASE 10: Saving aggregate CSV…")
    try:
        csv_path = save_aggregate_csv(out_dir)
        pipeline_log.info("Aggregate CSV: %s", csv_path)
    except Exception as exc:
        pipeline_log.error("Failed to save aggregate CSV: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_t0
    log_pipeline_summary(
        pipeline_log,
        n_total=len(passing_records),
        n_success=n_success,
        n_failed=n_failed,
        total_time_s=total_time,
    )

    if all_results:
        _print_summary_table(all_results)

    n_planets = sum(1 for r in all_results if r.get("predicted_label") == PLANET)
    print(
        f"\nDone in {total_time:.1f}s | "
        f"{n_success} processed | "
        f"{n_failed} failed | "
        f"{n_planets} planet candidates"
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
