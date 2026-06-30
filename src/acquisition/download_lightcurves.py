"""
src/acquisition/download_lightcurves.py
========================================
Downloads TESS SPOC short-cadence light curves from MAST for a given sector.

Uses ``lightkurve`` to search and download light curves, saves each target's
FITS file to ``data/raw/lightcurves/tic{TIC_ID}/``, and writes a manifest CSV
tracking download status and metadata.

Usage
-----
.. code-block:: bash

    python src/acquisition/download_lightcurves.py \\
        --sector 1 \\
        --max-targets 200 \\
        --cadence short \\
        --output-dir data/raw/lightcurves

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

# ---------------------------------------------------------------------------
# Ensure project src/ is importable regardless of CWD
# ---------------------------------------------------------------------------
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
    from astroquery.mast import Observations
except ImportError as _exc:
    logger.error("astroquery is not installed: %s", _exc)
    Observations = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_COLUMNS = [
    "tic_id",
    "sector",
    "filename",
    "download_time",
    "status",
    "filesize_bytes",
    "checksum_md5",
]

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds


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


def _get_tic_ids_for_sector(sector: int, max_targets: int) -> list[int]:
    """Return TIC IDs observed in *sector* by querying MAST via astroquery.

    Parameters
    ----------
    sector : int
        TESS sector number.
    max_targets : int
        Upper limit on returned TIC IDs.

    Returns
    -------
    list[int]
        Sorted unique TIC IDs (truncated to *max_targets*).
    """
    if Observations is None:
        raise RuntimeError("astroquery is required but not installed.")

    logger.info("Querying MAST for TIC IDs in sector %d ...", sector)
    try:
        obs_table = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            dataproduct_type="timeseries",
            calib_level=3,
        )
        if obs_table is None or len(obs_table) == 0:
            logger.warning("No MAST observations returned for sector %d.", sector)
            return []

        tic_ids: list[int] = []
        seen: set[int] = set()
        for row in obs_table:
            try:
                name: str = str(row["target_name"])
                if name.upper().startswith("TIC"):
                    tid = int(name.split()[-1])
                else:
                    tid = int(name)
                if tid not in seen:
                    seen.add(tid)
                    tic_ids.append(tid)
            except (ValueError, KeyError):
                continue

        tic_ids = sorted(tic_ids)[:max_targets]
        logger.info("Found %d unique TIC IDs in sector %d.", len(tic_ids), sector)
        return tic_ids

    except Exception as exc:
        logger.error("Failed to query MAST for sector %d: %s", sector, exc)
        return []


def _download_single(
    tic_id: int,
    sector: int,
    cadence: str,
    output_dir: Path,
) -> dict:
    """Download the light curve for one TIC target with retry logic.

    Parameters
    ----------
    tic_id : int
        TIC identifier.
    sector : int
        TESS sector number.
    cadence : str
        ``'short'``, ``'fast'``, or ``'long'``.
    output_dir : Path
        Root directory; files land in ``output_dir/tic{tic_id}/``.

    Returns
    -------
    dict
        Manifest row with keys matching :data:`MANIFEST_COLUMNS`.
    """
    target_dir = output_dir / f"tic{tic_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    row: dict = {
        "tic_id": tic_id,
        "sector": sector,
        "filename": "",
        "download_time": datetime.now(timezone.utc).isoformat(),
        "status": "FAILED",
        "filesize_bytes": 0,
        "checksum_md5": "",
    }

    exptime_map = {"short": 120, "fast": 20, "long": 1800}
    exptime = exptime_map.get(cadence.lower(), 120)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("TIC %d - searching (attempt %d/%d) ...", tic_id, attempt, MAX_RETRIES)
            search_result = lk.search_lightcurve(
                f"TIC {tic_id}",
                mission="TESS",
                sector=sector,
                exptime=exptime,
                author="SPOC",
            )
            if len(search_result) == 0:
                logger.debug("TIC %d - no SPOC LC in sector %d.", tic_id, sector)
                row["status"] = "NOT_FOUND"
                return row

            lc_collection = search_result.download_all(download_dir=str(target_dir))
            if lc_collection is None or len(lc_collection) == 0:
                row["status"] = "DOWNLOAD_EMPTY"
                return row

            fits_path: Optional[Path] = None
            for fits_file in target_dir.rglob("*.fits"):
                fits_path = fits_file
                break

            if fits_path is None:
                row["status"] = "FILE_NOT_FOUND_AFTER_DOWNLOAD"
                return row

            row["filename"] = str(fits_path.relative_to(output_dir.parent))
            row["filesize_bytes"] = fits_path.stat().st_size
            row["checksum_md5"] = _md5(fits_path)
            row["download_time"] = datetime.now(timezone.utc).isoformat()
            row["status"] = "OK"
            logger.debug("TIC %d - OK -> %s", tic_id, fits_path.name)
            return row

        except Exception as exc:
            logger.warning("TIC %d - attempt %d failed: %s", tic_id, attempt, exc)
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.debug("TIC %d - retrying in %d s ...", tic_id, sleep_time)
                time.sleep(sleep_time)

    row["status"] = "FAILED_AFTER_RETRIES"
    return row


def _write_manifest(manifest_rows: list[dict], manifest_path: Path) -> None:
    """Append *manifest_rows* to *manifest_path* (creates header if new).

    Parameters
    ----------
    manifest_rows : list[dict]
        Rows to append.
    manifest_path : Path
        Destination CSV file.
    """
    write_header = not manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(manifest_rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_sector(
    sector: int,
    max_targets: int = 500,
    cadence: str = "short",
    output_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> list[dict]:
    """Download TESS SPOC light curves for every target in a sector.

    Parameters
    ----------
    sector : int
        TESS sector number.
    max_targets : int
        Cap on the number of targets to download.
    cadence : str
        ``'short'`` (2-min), ``'fast'`` (20-sec), or ``'long'`` (30-min).
    output_dir : Path, optional
        Root download directory.  Defaults to ``data/raw/lightcurves/``.
    manifest_path : Path, optional
        Path for the manifest CSV.

    Returns
    -------
    list[dict]
        All manifest rows for this run.

    Raises
    ------
    RuntimeError
        If ``lightkurve`` is not installed.
    """
    if lk is None:
        raise RuntimeError("lightkurve is required but not installed.")

    root = project_root()
    if output_dir is None:
        output_dir = root / get("data.raw_dir", "data/raw") / "lightcurves"
    if manifest_path is None:
        manifest_path = Path(output_dir) / "manifest.csv"

    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)

    logger.info(
        "Starting download - sector=%d, max_targets=%d, cadence=%s",
        sector, max_targets, cadence,
    )

    tic_ids = _get_tic_ids_for_sector(sector, max_targets)
    if not tic_ids:
        logger.warning("No TIC IDs retrieved. Exiting.")
        return []

    manifest_rows: list[dict] = []
    status_counts: dict[str, int] = {}

    for tic_id in tqdm(tic_ids, desc=f"Sector {sector}", unit="target"):
        row = _download_single(tic_id, sector, cadence, output_dir)
        manifest_rows.append(row)
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    _write_manifest(manifest_rows, manifest_path)
    logger.info("Download complete. Status summary: %s", status_counts)
    logger.info("Manifest written to %s", manifest_path)
    return manifest_rows


def run(*args, **kwargs) -> list[dict]:
    """Orchestrator entry point mapping to download_sector."""
    sector = kwargs.get("sector", 1)
    max_targets = kwargs.get("max_targets")
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        output_dir = Path(output_dir)
    return download_sector(
        sector=sector,
        max_targets=max_targets if max_targets is not None else 500,
        output_dir=output_dir
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download TESS SPOC light curves from MAST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sector", type=int, required=True,
        help="TESS sector number to download.",
    )
    parser.add_argument(
        "--max-targets", type=int, default=500,
        help="Maximum number of targets to download.",
    )
    parser.add_argument(
        "--cadence", type=str, default="short",
        choices=["short", "fast", "long"],
        help="Light curve cadence type.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Root directory for downloaded FITS files.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configs/config.yaml (auto-detected if omitted).",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    try:
        load_config(_args.config)
    except Exception as _e:
        logger.warning("Could not load config: %s. Using built-in defaults.", _e)

    _output_dir = Path(_args.output_dir) if _args.output_dir else None

    _rows = download_sector(
        sector=_args.sector,
        max_targets=_args.max_targets,
        cadence=_args.cadence,
        output_dir=_output_dir,
    )

    _ok = sum(1 for r in _rows if r["status"] == "OK")
    print(f"\nDownloaded {_ok}/{len(_rows)} light curves successfully.")
