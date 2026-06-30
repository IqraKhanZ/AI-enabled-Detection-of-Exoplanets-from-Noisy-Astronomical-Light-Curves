"""
src/visualization/lightcurve_viewer.py
======================================
Standalone light curve viewer plotting raw, denoised/detrended, and phase-folded views.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

def plot_lightcurve(
    tic_id: int | str,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    detrended_flux: Optional[np.ndarray] = None,
    phase_result: Optional[Any] = None,
    transit_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    show: bool = False
) -> plt.Figure:
    """Create a 3-panel figure showing raw, detrended, and phase-folded light curves."""
    plt.style.use("dark_background")
    
    n_panels = 1
    if detrended_flux is not None:
        n_panels += 1
    if phase_result is not None:
        n_panels += 1
        
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]
        
    ax_idx = 0
    
    # 1. Raw Light Curve
    ax_raw = axes[ax_idx]
    ax_raw.scatter(time, flux, s=1, color="white", alpha=0.5, label="Raw SAP Flux")
    if transit_mask is not None:
        # Shade in-transit regions
        ax_raw.fill_between(time, np.min(flux), np.max(flux), where=transit_mask, 
                            color="red", alpha=0.15, label="Transit Mask")
    ax_raw.set_ylabel("SAP Flux")
    ax_raw.set_xlabel("Time (Days)")
    ax_raw.legend(loc="upper right")
    ax_raw.set_title(title or f"TIC {tic_id}")
    ax_idx += 1
    
    # 2. Detrended Light Curve
    if detrended_flux is not None:
        ax_det = axes[ax_idx]
        ax_det.scatter(time, detrended_flux, s=1, color="#1f77b4", alpha=0.6, label="Detrended Flux")
        ax_det.set_ylabel("Normalized Flux")
        ax_det.set_xlabel("Time (Days)")
        ax_det.legend(loc="upper right")
        ax_idx += 1
        
    # 3. Phase-Folded Light Curve
    if phase_result is not None:
        ax_fold = axes[ax_idx]
        
        # Check if phase_result has attributes or is dict
        phase = getattr(phase_result, "phase", None)
        global_view = getattr(phase_result, "global_view", None)
        local_view = getattr(phase_result, "local_view", None)
        
        if phase is not None and detrended_flux is not None:
            ax_fold.scatter(phase, detrended_flux, s=1, color="gray", alpha=0.3, label="Folded Data")
            
        if global_view is not None:
            # Reconstruct phase grid for global view [-0.5, 0.5]
            x_grid = np.linspace(-0.5, 0.5, len(global_view))
            ax_fold.plot(x_grid, global_view, color="#ff7f0e", linewidth=2, label="Global View Binned")
            
        ax_fold.set_ylabel("Normalized Flux")
        ax_fold.set_xlabel("Phase")
        ax_fold.legend(loc="upper right")
        
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved light curve plot to %s", save_path)
        
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot light curves for a target.")
    parser.add_argument("--tic-id", type=str, required=True, help="TIC ID of target.")
    parser.add_argument("--save-path", type=str, default=None, help="Output path for the plot.")
    args = parser.parse_args()
    
    # Generate dummy data for testing CLI
    time = np.linspace(0, 10, 1000)
    flux = 1.0 + np.random.normal(0, 1e-3, 1000)
    save_path = args.save_path or f"reports/tic_{args.tic_id}_plot.png"
    
    plot_lightcurve(
        tic_id=args.tic_id,
        time=time,
        flux=flux,
        detrended_flux=flux,
        save_path=save_path,
        show=False
    )

if __name__ == "__main__":
    main()
