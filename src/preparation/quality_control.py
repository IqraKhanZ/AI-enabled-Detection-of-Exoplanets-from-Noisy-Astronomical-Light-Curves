"""
src/preparation/quality_control.py
==================================
Flags and excludes unusable light curves based on quality-control rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightkurve as lk

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def check(fits_path: str | Path, max_nan: float = 0.20, min_days: float = 10.0) -> bool:
    """Check if a single light curve file passes QC criteria."""
    try:
        lc = lk.read(str(fits_path))
        flux = lc.flux.value
        time = lc.time.value
        if len(flux) == 0:
            return False
        nan_count = np.sum(np.isnan(flux))
        nan_frac = float(nan_count / len(flux))
        if nan_frac > max_nan:
            return False
        finite_mask = np.isfinite(flux) & np.isfinite(time)
        time_finite = time[finite_mask]
        if len(time_finite) <= 1:
            return False
        duration = float(np.max(time_finite) - np.min(time_finite))
        if duration < min_days:
            return False
        return True
    except Exception as exc:
        logger.debug("QC check failed for %s: %s", fits_path, exc)
        return False

def run_qc(
    lc_dir: str | Path,
    output_csv: str | Path,
    max_nan_frac: float = 0.20,
    min_duration: float = 10.0,
    max_crowding: float = 0.5,
    min_std: float = 1e-5
) -> None:
    """Run quality control checks on all light curves in the raw directory."""
    lc_dir = Path(lc_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fits_files = sorted(lc_dir.glob("**/*.fits")) + sorted(lc_dir.glob("**/*.fit"))
    if not fits_files:
        logger.warning("No light curve FITS files found in %s", lc_dir)
        # Create empty QC report to not break downstream steps
        df_empty = pd.DataFrame(columns=[
            "tic_id", "pass_qc", "nan_fraction", "time_baseline_days", 
            "flux_std", "crowding_metric", "fail_reasons"
        ])
        df_empty.to_csv(output_csv, index=False)
        return

    records = []
    for fp in fits_files:
        # Extract TIC ID from filename
        stem = fp.stem
        parts = stem.split("-")
        tic_id = None
        for part in parts:
            digits = "".join(c for c in part if c.isdigit())
            if 6 <= len(digits) <= 20:
                tic_id = int(digits)
                break
        
        if tic_id is None:
            continue

        pass_qc = True
        fail_reasons = []
        nan_frac = 1.0
        duration = 0.0
        flux_std = 0.0
        crowding = 1.0 # Default to fully crowded (fail) if metadata missing and check is strict

        try:
            # Read FITS using lightkurve
            lc = lk.read(str(fp))
            flux = lc.flux.value
            time = lc.time.value
            
            # 1. Check NaN fraction
            total_points = len(flux)
            if total_points > 0:
                nan_count = np.sum(np.isnan(flux))
                nan_frac = float(nan_count / total_points)
                if nan_frac > max_nan_frac:
                    pass_qc = False
                    fail_reasons.append(f"nan_fraction_{nan_frac:.3f}_gt_{max_nan_frac}")
            else:
                pass_qc = False
                fail_reasons.append("zero_data_points")
            
            # Filter finite for subsequent checks
            finite_mask = np.isfinite(flux) & np.isfinite(time)
            flux_finite = flux[finite_mask]
            time_finite = time[finite_mask]
            
            # 2. Check duration
            if len(time_finite) > 1:
                duration = float(np.max(time_finite) - np.min(time_finite))
                if duration < min_duration:
                    pass_qc = False
                    fail_reasons.append(f"duration_{duration:.2f}_lt_{min_duration}")
            else:
                pass_qc = False
                fail_reasons.append("insufficient_finite_points")

            # 3. Check std deviation (make sure it's not dead flat or zero)
            if len(flux_finite) > 1:
                # Normalize std relative to median
                med = np.median(flux_finite)
                if med != 0:
                    flux_std = float(np.std(flux_finite) / med)
                else:
                    flux_std = float(np.std(flux_finite))
                    
                if flux_std < min_std:
                    pass_qc = False
                    fail_reasons.append(f"std_{flux_std:.2e}_lt_{min_std}")
            else:
                pass_qc = False
                fail_reasons.append("cannot_compute_std")

            # 4. Check crowding metric from header if available
            # SPOC files store crowding in CROWDSAP header or column metadata
            crowding = 1.0
            if hasattr(lc, "meta") and "CROWDSAP" in lc.meta:
                crowding = float(lc.meta["CROWDSAP"])
            elif hasattr(lc, "meta") and "crowdsap" in lc.meta:
                crowding = float(lc.meta["crowdsap"])
            else:
                # Fallback check header directly from FITS HDU 1
                from astropy.io import fits
                with fits.open(str(fp)) as hdul:
                    if len(hdul) > 1 and "CROWDSAP" in hdul[1].header:
                        crowding = float(hdul[1].header["CROWDSAP"])
                    else:
                        crowding = 1.0 # default to 1.0 (no crowding issue) if key is missing

            if crowding < (1.0 - max_crowding): # CROWDSAP is fraction of target flux in aperture (1.0 = clean, 0.0 = fully crowded by other stars)
                # So if CROWDSAP < 0.5, it means > 50% contamination, which is bad.
                # Let's say crowding_metric in config is max_crowding_contamination = 0.5, so CROWDSAP must be >= 0.5
                pass_qc = False
                fail_reasons.append(f"crowdsap_{crowding:.3f}_lt_{1.0 - max_crowding}")

        except Exception as exc:
            pass_qc = False
            fail_reasons.append(f"exception_{type(exc).__name__}")
            logger.warning("QC failed to process TIC %d: %s", tic_id, exc)

        records.append({
            "tic_id": tic_id,
            "pass_qc": pass_qc,
            "nan_fraction": nan_frac,
            "time_baseline_days": duration,
            "flux_std": flux_std,
            "crowding_metric": crowding,
            "fail_reasons": ";".join(fail_reasons) if fail_reasons else "none"
        })

    df_qc = pd.DataFrame(records)
    df_qc.to_csv(output_csv, index=False)
    
    passed_count = df_qc["pass_qc"].sum()
    total_count = len(df_qc)
    logger.info("Quality Control completed. Saved to %s", output_csv)
    logger.info("QC stats: Passed %d / %d (%.2f%%)", 
                passed_count, total_count, (passed_count / total_count * 100) if total_count > 0 else 0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality control checks on TESS light curves.")
    parser.add_argument("--lc-dir", type=str, default=None, help="Path to raw light curves directory.")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to output QC flags CSV.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    args = parser.parse_args()
    
    root = project_root()
    lc_dir = args.lc_dir or root / get("paths.raw_lc", "data/raw/lightcurves", args.config)
    output_csv = args.output_csv or root / get("paths.processed", "data/processed", args.config) / "qc_flags.csv"
    
    max_nan_frac = get("quality_control.max_nan_fraction", 0.20, args.config)
    min_duration = get("quality_control.min_duration_days", 10.0, args.config)
    max_crowding = get("quality_control.max_crowding_metric", 0.5, args.config)
    min_std = get("quality_control.min_std_flux", 1e-5, args.config)
    
    run_qc(
        lc_dir=lc_dir,
        output_csv=output_csv,
        max_nan_frac=max_nan_frac,
        min_duration=min_duration,
        max_crowding=max_crowding,
        min_std=min_std
    )

if __name__ == "__main__":
    main()
