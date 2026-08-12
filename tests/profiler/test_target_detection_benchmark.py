"""
Benchmark test suite for Target Detection accuracy across datasets in catalog.yaml.
"""

import os
import yaml
import pandas as pd
import pytest

from backend.app.core.profiler.layer1_stats import profile_column
from backend.app.core.profiler.orchestrator import detect_target
from backend.app.core.ingestion.parsers import _detect_csv_separator


def load_benchmark_cases():
    catalog_path = "data/dataset/catalog.yaml"
    if not os.path.exists(catalog_path):
        return []
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    cases = []
    for ds in catalog.get("datasets", []):
        if ds.get("status") in ("verified", "pending") and os.path.exists(ds["file"]):
            cases.append((
                ds["id"],
                ds["file"],
                ds["target"],
                ds.get("separator", ","),
                ds["name"],
            ))
    return cases


BENCHMARK_CASES = load_benchmark_cases()


@pytest.mark.parametrize("ds_id,filepath,expected_target,sep,name", BENCHMARK_CASES)
def test_target_detection_accuracy(ds_id, filepath, expected_target, sep, name):
    with open(filepath, "rb") as f:
        content = f.read(100000)

    auto_sep = _detect_csv_separator(content)
    df = pd.read_csv(filepath, sep=auto_sep, nrows=2000)

    profiles = [profile_column(df[col]) for col in df.columns]
    target_analysis = detect_target(profiles, df)

    suggested = target_analysis.recommended_target
    candidates = [c.name for c in target_analysis.candidate_targets]

    assert suggested == expected_target, (
        f"[{ds_id} - {name}] Target detection failed! Expected '{expected_target}', "
        f"got suggested='{suggested}', candidates={candidates}"
    )
