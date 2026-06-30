"""
src/acquisition/integrity_check.py
=====================================
Verifies the integrity of all downloaded TESS light curve FITS files.

Reads the download manifest, checks each file for existence, readability,
expected HDU structure, minimum cadence count, and computes SHA-256 checksums.
Outputs a per-file integrity report CSV and prints a summary.

Status codes
------------
* OK       – file passes all checks
* MISSING  – file not found on disk
* CORRUPT  – file cannot be opened by astropy.io.fits
* INCOMPLETE – fewer than ``--min-cadences`` time steps found

Usage
-----
.. code-block:: bash

    python src/acquisition/integrity_check.py \\
        --manifest data/raw/lightcurves/manifest.csv \\
        --output   data/raw/integrity_report.csv \\
        --min-cadences 100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config import get, load_config, project_root  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

try:
    from astropy.io import fits
except ImportError as _exc:
    logger.error("astropy is not installed: %s", _exc)
    fits = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_COLUMNS = [
    "tic_id",
    "filename",
    "status",
    "checksum_sha256",
    "n_cadences",
    "filesize_bytes",
    "message",
]

EXPECTED_EXTNAME = "LIGHTCURVE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Compute SHA-256 hex-digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_file(
    tic_id: object,
    filename: str,
    data_root: Path,
    min_cadences: int,
) -> dict:
    """Run all integrity checks on one FITS file.

    Parameters
    ----------
    tic_id : object
        TIC identifier (for reporting).
    filename : str
        Relative path stored in the manifest.
    data_root : Path
        Root directory that *filename* is relative to.
    min_cadences : int
        Minimum acceptable number of time steps.

    Returns
    -------
    dict
        A report row with keys matching :data:`REPORT_COLUMNS`.
    """
    row: dict = {
        "tic_id": tic_id,
        "filename": filename,
        "status": "UNKNOWN",
        "checksum_sha256": "",
        "n_cadences": 0,
        "filesize_bytes": 0,
        "message": "",
    }

    # Resolve path
    fpath = data_root / filename if filename else None
    if fpath is None or not filename:
        row["status"] = "MISSING"
        row["message"] = "Empty filename in manifest."
        return row

    if not fpath.exists():
        row["status"] = "MISSING"
        row["message"] = f"File not found: {fpath}"
        return row

    row["filesize_bytes"] = fpath.stat().st_size

    # SHA-256
    try:
        row["checksum_sha256"] = _sha256(fpath)
    except OSError as exc:
        row["status"] = "CORRUPT"
        row["message"] = f"Cannot read file for checksum: {exc}"
        return row

    # FITS validation
    if fits is None:
        row["status"] = "UNCHECKED"
        row["message"] = "astropy not installed; FITS check skipped."
        return row

    try:
        with fits.open(str(fpath), memmap=False) as hdul:
            ext_names = [h.name for h in hdul]

            # Check for LIGHTCURVE extension
            if EXPECTED_EXTNAME not in ext_names:
                row["status"] = "CORRUPT"
                row["message"] = (
                    f"Expected HDU '{EXPECTED_EXTNAME}' not found. "
                    f"Extensions: {ext_names}"
                )
                return row

            lc_hdu = hdul[EXPECTED_EXTNAME]
            # Count cadences via TIME column
            n_cad = 0
            if hasattr(lc_hdu, "data") and lc_hdu.data is not None:
                if "TIME" in lc_hdu.columns.names:
                    n_cad = len(lc_hdu.data["TIME"])
                else:
                    n_cad = len(lc_hdu.data)

            row["n_cadences"] = n_cad

            if n_cad < min_cadences:
                row["status"] = "INCOMPLETE"
                row["message"] = (
                    f"Only {n_cad} cadences found; minimum is {min_cadences}."
                )
                return row

    except Exception as exc:
        row["status"] = "CORRUPT"
        row["message"] = f"FITS open error: {exc}"
        return row

    row["status"] = "OK"
    row["message"] = ""
    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_integrity_check(
    manifest_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    min_cadences: int = 100,
) -> pd.DataFrame:
    """Check all files listed in the manifest and write a report.

    Parameters
    ----------
    manifest_path : Path, optional
        Path to ``manifest.csv``.  Defaults to
        ``data/raw/lightcurves/manifest.csv``.
    output_path : Path, optional
        Destination for the integrity report CSV.  Defaults to
        ``data/raw/integrity_report.csv``.
    min_cadences : int
        Minimum acceptable cadence count.

    Returns
    -------
    pd.DataFrame
        Integrity report dataframe.
    """
    root = project_root()
    raw_dir = root / get("data.raw_dir", "data/raw")

    if manifest_path is None:
        manifest_path = raw_dir / "lightcurves" / "manifest.csv"
    if output_path is None:
        output_path = raw_dir / "integrity_report.csv"

    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, low_memory=False)
    logger.info("Loaded manifest: %d rows from %s", len(manifest), manifest_path)

    # Data root is the parent of the first path component stored in 'filename'
    # Typically filenames are relative to data/raw/
    data_root = raw_dir

    report_rows: list[dict] = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Integrity check"):
        tic_id = row.get("tic_id", "UNKNOWN")
        filename = str(row.get("filename", ""))
        result = _check_file(tic_id, filename, data_root, min_cadences)
        report_rows.append(result)

    report_df = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    logger.info("Integrity report written to %s", output_path)

    # Summary
    counts = report_df["status"].value_counts()
    print("\n=== Integrity Check Summary ===")
    for status, n in counts.items():
        print(f"  {status:20s}: {n:>6d}")
    print(f"  {'TOTAL':20s}: {len(report_df):>6d}")
    ok_frac = counts.get("OK", 0) / max(len(report_df), 1) * 100
    print(f"\n  Pass rate: {ok_frac:.1f}%")

    return report_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify integrity of downloaded TESS FITS light curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to manifest.csv.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write integrity_report.csv.",
    )
    parser.add_argument(
        "--min-cadences", type=int, default=100,
        help="Minimum number of cadences required to pass.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configs/config.yaml.",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    try:
        load_config(_args.config)
    except Exception as _e:
        logger.warning("Could not load config: %s. Using built-in defaults.", _e)

    _manifest = Path(_args.manifest) if _args.manifest else None
    _output = Path(_args.output) if _args.output else None

    _report = run_integrity_check(
        manifest_path=_manifest,
        output_path=_output,
        min_cadences=_args.min_cadences,
    )
