import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Tuple

from .column_profile import ColumnProfile, DataRole
from .layer1_stats import profile_column
from .layer2_semantic import detect_semantic
from .layer3_llm import refine_with_llm
from ..config import (
    ENTROPY_LIMIT_LOW,
    ENTROPY_LIMIT_MED,
    ENTROPY_LIMIT_HIGH,
    ENTROPY_SCORE_LOW,
    ENTROPY_SCORE_MED,
    ENTROPY_SCORE_HIGH,
)
from .target_analysis import (
    TargetAnalysis,
    CandidateTarget,
    TargetSignals,
    TargetRole,
    ChurnColumnGroupItem,
    GroupRole,
)

# Recipe Table Mapping (Phase 1 Goal 2)
ROLE_RECIPES = {
    DataRole.ID: {"impute": "drop", "transform": "passthrough"},
    DataRole.TARGET: {"impute": "drop_row", "transform": "label"},
    DataRole.NUMERIC: {"impute": "median", "transform": "standard"},
    DataRole.CATEGORICAL: {"impute": "mode", "transform": "ohe"},
    DataRole.DATETIME: {"impute": "median", "transform": "date_parts"},
    DataRole.TEXT: {"impute": "constant", "transform": "tfidf"},
    DataRole.IGNORE: {"impute": "drop", "transform": "drop"},
}


def _cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return 0
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    if min((kcorr - 1), (rcorr - 1)) <= 0:
        return 0
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _calculate_correlation(
    target_col: str, col_name: str, dtype: str, unique_count: int, df: pd.DataFrame
) -> float:
    if target_col == col_name:
        return 1.0
    y = df[target_col].dropna()
    x = df[col_name].dropna()
    common_idx = x.index.intersection(y.index)
    x_common, y_common = x.loc[common_idx], y.loc[common_idx]

    if len(x_common) < 2 or x_common.nunique() <= 1:
        return 0.0

    try:
        if dtype in ["float64", "int64"] and unique_count > 10:
            y_encoded = pd.factorize(y_common)[0]
            corr, _ = stats.pointbiserialr(y_encoded, x_common)
            return float(abs(corr)) if not np.isnan(corr) else 0.0
        else:
            return float(_cramers_v(x_common, y_common))
    except Exception:
        return 0.0


def detect_target(profiles: list[dict], df: pd.DataFrame) -> TargetAnalysis:
    candidate_list = []
    total_cols = len(df.columns)

    for i, p in enumerate(profiles):
        col = p["name"]
        unique_count = p.get("unique_count", 0)
        if unique_count < 2:
            continue

        score = 0.0

        # Primary Signals
        is_binary = unique_count == 2
        if is_binary:
            score += 1.0
        elif 2 < unique_count <= 5:
            score += 0.5

        ent = p.get("entropy", 0.0)
        ent_score = 0.0
        if 0 < ent <= ENTROPY_LIMIT_LOW:
            ent_score = ENTROPY_SCORE_LOW
        elif ENTROPY_LIMIT_LOW < ent <= ENTROPY_LIMIT_MED:
            ent_score = ENTROPY_SCORE_MED
        elif ENTROPY_LIMIT_MED < ent <= ENTROPY_LIMIT_HIGH:
            ent_score = ENTROPY_SCORE_HIGH
        score += ent_score

        # Secondary Signals
        position_bonus = 0.0
        if i >= total_cols - 2:
            position_bonus = 0.1
            score += position_bonus

        name_lower = col.lower()
        keyword_match = False
        if any(
            kw in name_lower
            for kw in ["target", "label", "churn", "status", "attrition"]
        ):
            keyword_match = True
            score += 0.1

        candidate_list.append(
            {
                "name": col,
                "score": score,
                "idx": i,
                "signals": {
                    "is_binary": is_binary,
                    "entropy": ent,
                    "entropy_score": ent_score,
                    "keyword_match": keyword_match,
                    "position_bonus": position_bonus,
                },
            }
        )

    # Sort candidates by score descending, then by original column order ascending
    candidates_sorted = sorted(candidate_list, key=lambda x: (-x["score"], x["idx"]))

    recommended_target = ""
    if candidates_sorted:
        top_cand = candidates_sorted[0]
        if top_cand["score"] >= 1.0:
            recommended_target = top_cand["name"]

    candidate_targets = []
    churn_column_group = []
    recommended_auxiliary = []
    leakage_suspects = []

    if recommended_target:
        # Pre-calculate leakage mapping for group roles classification
        temp_leakage = {}
        y = df[recommended_target].dropna()
        for p in profiles:
            col = p["name"]
            if col == recommended_target or p.get("inferred_role") in [
                DataRole.ID,
                DataRole.IGNORE,
            ]:
                continue
            corr = _calculate_correlation(
                recommended_target,
                col,
                p.get("inferred_dtype", "object"),
                p.get("unique_count", 0),
                df,
            )
            if corr > 0.95:
                temp_leakage[col] = corr

        # 1. churn_column_group
        churn_column_group.append(
            ChurnColumnGroupItem(
                name=recommended_target,
                correlation_with_target=1.0,
                group_role=GroupRole.PRIMARY,
            )
        )

        for p in profiles:
            col = p["name"]
            if col == recommended_target:
                continue
            corr = _calculate_correlation(
                recommended_target,
                col,
                p.get("inferred_dtype", "object"),
                p.get("unique_count", 0),
                df,
            )

            if corr >= 0.5:
                group_role = GroupRole.AUXILIARY
                if corr > 0.98:
                    group_role = GroupRole.DUPLICATE
                elif col in temp_leakage:
                    group_role = GroupRole.LEAKAGE_SUSPECT

                churn_column_group.append(
                    ChurnColumnGroupItem(
                        name=col, correlation_with_target=corr, group_role=group_role
                    )
                )

                if group_role == GroupRole.AUXILIARY:
                    recommended_auxiliary.append(col)
                elif group_role == GroupRole.LEAKAGE_SUSPECT:
                    leakage_suspects.append(col)

        # 2. candidate_targets (take top 3 candidates)
        top_candidates = candidates_sorted[:3]
        for rank_idx, cand in enumerate(top_candidates):
            col = cand["name"]
            suggested_role = TargetRole.AUXILIARY
            if col == recommended_target:
                suggested_role = TargetRole.TARGET
            else:
                p_col = next(p for p in profiles if p["name"] == col)
                corr = _calculate_correlation(
                    recommended_target,
                    col,
                    p_col.get("inferred_dtype", "object"),
                    p_col.get("unique_count", 0),
                    df,
                )
                if corr > 0.98:
                    suggested_role = TargetRole.DUPLICATE
                elif col in temp_leakage:
                    suggested_role = TargetRole.LEAKAGE

            candidate_targets.append(
                CandidateTarget(
                    name=col,
                    rank=rank_idx + 1,
                    score=cand["score"],
                    signals=TargetSignals(**cand["signals"]),
                    suggested_role=suggested_role,
                )
            )

    return TargetAnalysis(
        recommended_target=recommended_target,
        candidate_targets=candidate_targets,
        churn_column_group=churn_column_group,
        recommended_auxiliary=recommended_auxiliary,
        leakage_suspects=leakage_suspects,
    )


def check_leakage(target_col: str, profiles: list[dict], df: pd.DataFrame):
    if not target_col or target_col not in df.columns:
        return

    for p in profiles:
        col = p["name"]
        if col == target_col or p["inferred_role"] in [DataRole.ID, DataRole.IGNORE]:
            continue

        corr = _calculate_correlation(
            target_col, col, p["inferred_dtype"], p["unique_count"], df
        )
        if corr > 0.95:
            p["potential_leakage"] = True
            p["leakage_score"] = float(corr)


def run_profiling(df: pd.DataFrame) -> Tuple[list[ColumnProfile], TargetAnalysis]:
    # Layer 1
    profiles_dict = [profile_column(df[col]) for col in df.columns]

    # Target Detection
    target_analysis = detect_target(profiles_dict, df)
    suggested_target = target_analysis.recommended_target

    if suggested_target:
        for p in profiles_dict:
            if p["name"] == suggested_target:
                p["inferred_role"] = DataRole.TARGET
                p["confidence_score"] = 1.0
                break

    # Leakage Check
    check_leakage(suggested_target, profiles_dict, df)

    from ..config import COMPOSITE_SYNTHESIS_ENABLED

    if COMPOSITE_SYNTHESIS_ENABLED and target_analysis.churn_column_group:
        from .target_synthesizer import synthesize_target

        composite_config, cpi_series = synthesize_target(
            df, target_analysis.churn_column_group, target_analysis.recommended_target
        )
        target_analysis.composite_target = composite_config
        # Attach CPI to df ONLY if auto-synthesized (<=2 aux cols)
        if cpi_series is not None:
            df[composite_config.cpi_column_name] = cpi_series

    final_profiles = []
    for p in profiles_dict:
        if (
            p["name"] == target_analysis.composite_target.cpi_column_name
            if (
                target_analysis.composite_target
                and target_analysis.composite_target.cpi_column_name in df.columns
            )
            else False
        ):
            # Skip profiling the synthesized CPI column if it got added to df
            continue
        if p["inferred_role"] != DataRole.TARGET:
            # Layer 2
            p = detect_semantic(df[p["name"]], p)
            # Layer 3
            sample_vals = (
                df[p["name"]].dropna().sample(min(5, df[p["name"]].count())).tolist()
                if df[p["name"]].count() > 0
                else []
            )
            p = refine_with_llm(sample_vals, p)

        # Assign Recipes (Phase 1 Goal 2)
        recipe = ROLE_RECIPES.get(p["inferred_role"], ROLE_RECIPES[DataRole.IGNORE])
        p["impute_strategy"] = recipe["impute"]
        p["transform_strategy"] = recipe["transform"]

        # Pydantic safety
        p.setdefault("regex_pattern", None)
        p.setdefault("mean_length", None)
        p.setdefault("leakage_score", None)
        p.setdefault("potential_leakage", False)

        final_profiles.append(ColumnProfile(**p))

    return final_profiles, target_analysis
