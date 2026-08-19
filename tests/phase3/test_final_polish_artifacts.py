"""
Tests for Final Polish Phase Artifacts and Empirical Integrity.
Verifies publication visual storytelling figures, statistical validation,
cross-dataset benchmarks, and manuscript completeness.
"""
import json
from pathlib import Path
import pytest


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent.parent


def test_visual_storytelling_figures_exist(project_root):
    """Verify that all 3 new visual storytelling figures exist in PDF and PNG (300 DPI)."""
    fig_dir = project_root / "docs" / "openscience" / "paper1" / "figures"
    assert fig_dir.exists(), "Figures directory does not exist"

    expected_figures = [
        "fig_overview_pipeline",
        "fig_radar_comprehensive",
        "fig_trivector_conflict",
        "fig_pareto_efficiency",
    ]

    for fig_name in expected_figures:
        pdf_path = fig_dir / f"{fig_name}.pdf"
        png_path = fig_dir / f"{fig_name}.png"

        assert pdf_path.exists(), f"Missing {pdf_path}"
        assert png_path.exists(), f"Missing {png_path}"
        assert pdf_path.stat().st_size > 1000, f"{pdf_path} is too small / corrupted"
        assert png_path.stat().st_size > 1000, f"{png_path} is too small / corrupted"


def test_appendix_figures_exist(project_root):
    """Verify that all 4 causal appendix figures exist in PDF and PNG."""
    fig_dir = project_root / "docs" / "openscience" / "paper1" / "figures"

    appendix_figures = [
        "fig_appendix_taylor_residual",
        "fig_appendix_effective_rank",
        "fig_appendix_gradient_conflict",
        "fig_appendix_coreset_gap",
    ]

    for fig_name in appendix_figures:
        pdf_path = fig_dir / f"{fig_name}.pdf"
        png_path = fig_dir / f"{fig_name}.png"

        assert pdf_path.exists(), f"Missing {pdf_path}"
        assert png_path.exists(), f"Missing {png_path}"
        assert pdf_path.stat().st_size > 1000, f"{pdf_path} is too small"
        assert png_path.stat().st_size > 1000, f"{png_path} is too small"


def test_gap_analysis_statistical_significance(project_root):
    """Verify statistical significance values in gap_analysis_results.json."""
    json_path = project_root / "docs" / "openscience" / "paper1" / "gap_analysis_results.json"
    assert json_path.exists(), f"Missing {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = data["statistical_significance"]
    assert stats["statistically_significant"] is True
    assert stats["paired_t_test_p_val"] < 0.01, "p-value not < 0.01"

    # Check computational profiling table
    profiling = data["computational_profiling"]
    assert len(profiling) == 5
    config_names = [p["configuration"] for p in profiling]
    assert "Full + PCGrad" in config_names

    # Check A-GEM baseline
    agem = data["agem_baseline"]
    assert agem["baseline"] == "A-GEM"
    assert 0.5 < agem["AUC_A"] < 1.0


def test_cross_dataset_validation_output(project_root):
    """Verify cross-dataset validation results on Bank Customer Churn."""
    json_path = project_root / "docs" / "openscience" / "paper1" / "cross_dataset_results.json"
    md_path = project_root / "docs" / "openscience" / "paper1" / "cross_dataset_results.md"

    assert json_path.exists(), f"Missing {json_path}"
    assert md_path.exists(), f"Missing {md_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_configs = [
        "Fine-tuning",
        "Replay-Only",
        "EWC-Only",
        "Full Naive (EWC+Replay)",
        "Full + PCGrad",
    ]

    for cfg in required_configs:
        assert cfg in data, f"Missing {cfg} in cross-dataset results"
        assert "AUC_A_final" in data[cfg]
        assert "AUC_B" in data[cfg]
        assert "forgetting_pct" in data[cfg]


def test_latex_and_markdown_consistency(project_root):
    """Verify that both LaTeX and Markdown manuscripts contain key spotlight elements."""
    tex_path = project_root / "docs" / "openscience" / "paper1" / "paper_continual_mtl_conflict.tex"
    md_path = project_root / "docs" / "openscience" / "paper1" / "paper_continual_mtl_conflict.md"

    assert tex_path.exists(), f"Missing {tex_path}"
    assert md_path.exists(), f"Missing {md_path}"

    tex_content = tex_path.read_text(encoding="utf-8")
    md_content = md_path.read_text(encoding="utf-8")

    key_phrases = [
        "Tri-Vector",
        "A-GEM",
        "Bank Customer Churn",
        "0.0003",
        "PCGrad",
        "Broader Impacts",
        "GDPR",
    ]

    for phrase in key_phrases:
        assert phrase in tex_content, f"LaTeX manuscript missing phrase: {phrase}"
        assert phrase in md_content, f"Markdown manuscript missing phrase: {phrase}"
