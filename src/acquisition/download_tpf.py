"""
src/acquisition/download_tpf.py
=================================
Downloads Target Pixel Files (TPFs) for centroid motion analysis.

For each TIC ID listed in the light-curve manifest, searches MAST for TESS
SPOC target pixel files using ``lightkurve.search_targetpixelfile()``,
downloads them to ``data/raw/tpf/tic{TIC_ID}/``, and writes a manifest CSV
at ``data/raw/tpf/tpf_manifest.csv``.

Missing TPFs are handled gracefully — not all targets have available TPF data.

Usage
-----
.. code-block:: bash

    python src/acquisition/download_tpf.py \\
        --manifest data/raw/lightcurves/manifest.csv \\
        --output-dir data/raw/tpf \\
        --sector 1 \\
        --cadence short
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime, timezone
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
    import lightkurve as lk
except ImportError as _exc:
    logger.error("lightkurve is not installed: %s", _exc)
    lk = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TPF_MANIFEST_COLUMNS = [
    "tic_id",
    "sector",
    "filename",
    "download_time",
    "status",
    "filesize_bytes",
    "checksum_md5",
    "n_pixels_x",
    "n_pixels_y",
]

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5(path: Path) -> str:
    """Compute MD5 hex-digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex-encoded MD5 digest.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _pixel_shape(fits_path: Path) -> tuple[int, int]:
    """Read the pixel dimensions from a TPF FITS file.

    Parameters
    ----------
    fits_path : Path
        Path to the TPF FITS file.

    Returns
    -------
    tuple[int, int]
        ``(n_cols, n_rows)`` pixel dimensions, or ``(0, 0)`` on failure.
    """
    try:
        from astropy.io import fits as afits
        with afits.open(str(fits_path), memmap=False) as hdul:
            for hdu in hdul:
                if hasattr(hdu, "data") and hdu.data is not None:
                    if hasattr(hdu.data, "dtype") and hdu.data.ndim >= 3:
                        # Shape: (time, row, col)
                        return int(hdu.data.shape[2]), int(hdu.data.shape[1])
    except Exception:
        pass
    return 0, 0


def _download_tpf_single(
    tic_id: int,
    sector: Optional[int],
    cadence: str,
    output_dir: Path,
) -> dict:
    """Download the TPF for one TIC target with retry logic.

    Parameters
    ----------
    tic_id : int
        TIC identifier.
    sector : int or None
        TESS sector; pass ``None`` to download all available sectors.
    cadence : str
        ``'short'``, ``'fast'``, or ``'long'``.
    output_dir : Path
        Root TPF directory; files land in ``output_dir/tic{tic_id}/``.

    Returns
    -------
    dict
        Manifest row with keys matching :data:`TPF_MANIFEST_COLUMNS`.
    """
    target_dir = output_dir / f"tic{tic_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    row: dict = {
        "tic_id": tic_id,
        "sector": sector if sector is not None else "",
        "filename": "",
        "download_time": datetime.now(timezone.utc).isoformat(),
        "status": "FAILED",
        "filesize_bytes": 0,
        "checksum_md5": "",
        "n_pixels_x": 0,
        "n_pixels_y": 0,
    }

    exptime_map = {"short": 120, "fast": 20, "long": 1800}
    exptime = exptime_map.get(cadence.lower(), 120)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("TIC %d – TPF search (attempt %d/%d)...", tic_id, attempt, MAX_RETRIES)

            search_kwargs: dict = {
                "mission": "TESS",
                "exptime": exptime,
                "author": "SPOC",
            }
            if sector is not None:
                search_kwargs["sector"] = sector

            sr = lk.search_targetpixelfile(f"TIC {tic_id}", **search_kwargs)

            if len(sr) == 0:
                logger.debug("TIC %d – no TPF found.", tic_id)
                row["status"] = "NOT_FOUND"
                return row

            tpf_collection = sr.download_all(download_dir=str(target_dir))
            if tpf_collection is None or len(tpf_collection) == 0:
                row["status"] = "DOWNLOAD_EMPTY"
                return row

            # Locate first FITS file
            fits_path: Optional[Path] = None
            for f in target_dir.rglob("*.fits"):
                fits_path = f
                break

            if fits_path is None:
                row["status"] = "FILE_NOT_FOUND_AFTER_DOWNLOAD"
                return row

            nx, ny = _pixel_shape(fits_path)

            row["filename"] = str(fits_path.relative_to(output_dir.parent))
            row["filesize_bytes"] = fits_path.stat().st_size
            row["checksum_md5"] = _md5(fits_path)
            row["download_time"] = datetime.now(timezone.utc).isoformat()
            row["status"] = "OK"
            row["n_pixels_x"] = nx
            row["n_pixels_y"] = ny

            logger.debug("TIC %d – TPF OK -> %s (%dx%d px)", tic_id, fits_path.name, nx, ny)
            return row

        except Exception as exc:
            logger.warning("TIC %d – TPF attempt %d failed: %s", tic_id, attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))

    row["status"] = "FAILED_AFTER_RETRIES"
    return row


def _write_tpf_manifest(rows: list[dict], manifest_path: Path) -> None:
    """Append *rows* to the TPF manifest CSV.

    Parameters
    ----------
    rows : list[dict]
        Rows to append.
    manifest_path : Path
        Destination CSV.
    """
    write_header = not manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TPF_MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_tpfs(
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sector: Optional[int] = None,
    cadence: str = "short",
) -> list[dict]:
    """Download TPFs for all TIC IDs listed in the manifest.

    Parameters
    ----------
    manifest_path : Path, optional
        Path to ``manifest.csv``.  Defaults to
        ``data/raw/lightcurves/manifest.csv``.
    output_dir : Path, optional
        Root TPF directory.  Defaults to ``data/raw/tpf/``.
    sector : int or None
        TESS sector to restrict to.  ``None`` downloads all available.
    cadence : str
        ``'short'``, ``'fast'``, or ``'long'``.

    Returns
    -------
    list[dict]
        All TPF manifest rows.

    Raises
    ------
    RuntimeError
        If lightkurve is not installed.
    FileNotFoundError
        If the manifest CSV does not exist.
    """
    if lk is None:
        raise RuntimeError("lightkurve is required but not installed.")

    root = project_root()
    if manifest_path is None:
        manifest_path = root / get("data.raw_dir", "data/raw") / "lightcurves" / "manifest.csv"
    if output_dir is None:
        output_dir = root / get("data.raw_dir", "data/raw") / "tpf"

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    tpf_manifest_path = output_dir / "tpf_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, low_memory=False)
    tic_ids = sorted(manifest["tic_id"].dropna().astype(int).unique().tolist())
    logger.info("Downloading TPFs for %d unique TIC IDs ...", len(tic_ids))

    all_rows: list[dict] = []
    status_counts: dict[str, int] = {}

    for tic_id in tqdm(tic_ids, desc="Downloading TPFs", unit="target"):
        row = _download_tpf_single(tic_id, sector, cadence, output_dir)
        all_rows.append(row)
        s = str(row["status"])
        status_counts[s] = status_counts.get(s, 0) + 1

    _write_tpf_manifest(all_rows, tpf_manifest_path)
    logger.info("TPF download complete. Status: %s", status_counts)
    logger.info("TPF manifest written to %s", tpf_manifest_path)

    ok = status_counts.get("OK", 0)
    not_found = status_counts.get("NOT_FOUND", 0)
    print(f"\nTPFs downloaded: {ok}/{len(tic_ids)}  (not found: {not_found})")
    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TESS Target Pixel Files from MAST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to light-curve manifest.csv.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Root directory for downloaded TPF FITS files.",
    )
    parser.add_argument(
        "--sector", type=int, default=None,
        help="TESS sector to restrict downloads to (default: all).",
    )
    parser.add_argument(
        "--cadence", type=str, default="short",
        choices=["short", "fast", "long"],
        help="Cadence type for TPF search.",
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
    _output_dir = Path(_args.output_dir) if _args.output_dir else None

    _rows = download_tpfs(
        manifest_path=_manifest,
        output_dir=_output_dir,
        sector=_args.sector,
        cadence=_args.cadence,
    )
    _ok = sum(1 for r in _rows if r["status"] == "OK")
    print(f"Finished. {_ok}/{len(_rows)} TPFs downloaded OK.")
