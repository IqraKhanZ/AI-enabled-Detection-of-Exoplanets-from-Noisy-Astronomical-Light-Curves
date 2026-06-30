"""
tests/test_visualization.py
===========================
Smoke tests to ensure visualization modules render figures without raising exceptions.
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for headless tests
import matplotlib.pyplot as plt
import numpy as np
import pytest

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_SRC_PKG_DIR = _SRC_DIR / "src"
if str(_SRC_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_PKG_DIR))

from visualization.lightcurve_viewer import plot_lightcurve

def test_plot_lightcurve_raw() -> None:
    time = np.linspace(0, 10, 100)
    flux = 1.0 + np.random.normal(0, 1e-3, 100)
    
    fig = plot_lightcurve(
        tic_id=12345,
        time=time,
        flux=flux,
        show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_lightcurve_detrended() -> None:
    time = np.linspace(0, 10, 100)
    flux = 1.0 + np.random.normal(0, 1e-3, 100)
    detrended = flux - 1.0
    
    fig = plot_lightcurve(
        tic_id=12345,
        time=time,
        flux=flux,
        detrended_flux=detrended,
        show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_lightcurve_phasefolded() -> None:
    time = np.linspace(0, 10, 100)
    flux = 1.0 + np.random.normal(0, 1e-3, 100)
    detrended = flux - 1.0
    
    class DummyPhaseResult:
        phase = np.linspace(-0.5, 0.5, 100)
        global_view = np.ones(200)
        local_view = np.ones(50)
        
    pr = DummyPhaseResult()
    
    fig = plot_lightcurve(
        tic_id=12345,
        time=time,
        flux=flux,
        detrended_flux=detrended,
        phase_result=pr,
        show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
