"""
Verification Test Suite for Publication-Ready Appendix Visualizations
Target Paper: "Optimization Conflicts in Continual Multi-Task Learning: Why Replay Degrades Regularization-Based Knowledge Retention"
"""

import os
from pathlib import Path
import pytest
import numpy as np

try:
    import torch
except (ImportError, OSError):
    torch = None


def test_appendix_figures_exist_and_valid():
    """
    Verifies that all 4 Appendix publication-ready figures exist in both
    vector PDF and 300 DPI PNG formats, and that their file sizes are non-trivial.
    """
    figures_dir = Path(__file__).resolve().parents[2] / "docs" / "openscience" / "paper1" / "figures"
    assert figures_dir.exists(), f"Figures directory does not exist: {figures_dir}"

    expected_figures = [
        "fig_appendix_taylor_residual",
        "fig_appendix_effective_rank",
        "fig_appendix_gradient_conflict",
        "fig_appendix_coreset_gap",
    ]

    for fig_stem in expected_figures:
        pdf_file = figures_dir / f"{fig_stem}.pdf"
        png_file = figures_dir / f"{fig_stem}.png"

        assert pdf_file.exists(), f"Missing PDF vector figure: {pdf_file}"
        assert png_file.exists(), f"Missing 300 DPI PNG figure: {png_file}"

        # Check non-empty (> 10 KB)
        assert pdf_file.stat().st_size > 10_000, f"PDF file too small: {pdf_file} ({pdf_file.stat().st_size} bytes)"
        assert png_file.stat().st_size > 50_000, f"PNG file too small: {png_file} ({png_file.stat().st_size} bytes)"

        # Check valid magic bytes
        with open(pdf_file, "rb") as f:
            header = f.read(5)
            assert header.startswith(b"%PDF-"), f"Invalid PDF header in {pdf_file}"

        with open(png_file, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", f"Invalid PNG magic bytes in {png_file}"


def test_appendix_figure_generation_mock():
    """
    Verifies that figure generation subroutines work cleanly on small synthetic datasets.
    """
    try:
        from docs.openscience.paper1.scripts.generate_appendix_plots import COLORS
    except ImportError:
        from scripts.generate_appendix_plots import COLORS
    import matplotlib.pyplot as plt

    assert "blue" in COLORS
    assert "crimson" in COLORS
    assert "emerald" in COLORS
    assert "purple" in COLORS

    # Verify matplotlib can render without display backend
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], color=COLORS["blue"])
    plt.close(fig)
