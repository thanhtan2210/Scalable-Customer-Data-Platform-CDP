import pandas as pd
import re
from .column_profile import DataRole

EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
URL_REGEX = r"^https?:\/\/"


def detect_semantic(series: pd.Series, profile: dict) -> dict:
    if profile["inferred_role"] in [
        DataRole.IGNORE,
        DataRole.DATETIME,
        DataRole.NUMERIC,
        DataRole.TARGET,
    ]:
        return profile

    clean_series = series.dropna().astype(str)
    if clean_series.empty:
        return profile

    mean_length = clean_series.str.len().mean()
    profile["mean_length"] = float(mean_length)

    # Pattern matching
    sample = clean_series.head(100)
    email_matches = sample.str.match(EMAIL_REGEX).mean()
    url_matches = sample.str.match(URL_REGEX).mean()

    if email_matches > 0.8 or url_matches > 0.8:
        profile["inferred_role"] = DataRole.IGNORE
        profile["regex_pattern"] = "email" if email_matches > 0.8 else "url"
        profile["confidence_score"] = 0.9
        return profile

    # Text length ratio (if strings are very long, it's free text)
    if mean_length > 50:
        profile["inferred_role"] = DataRole.TEXT
        profile["confidence_score"] = 0.8

    elif (
        profile["inferred_role"] != DataRole.ID
        and profile["unique_count"] <= 20
        and mean_length < 50
    ):
        profile["inferred_role"] = DataRole.CATEGORICAL
        profile["confidence_score"] = min(profile["confidence_score"] + 0.3, 1.0)

    return profile
