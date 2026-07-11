import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """Cuts off values at specified lower and upper percentiles."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower).to_dict()
        self.upper_bounds_ = X_df.quantile(self.upper).to_dict()
        self.feature_names_in_ = (
            X_df.columns.astype(str).tolist() if hasattr(X_df, "columns") else None
        )
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "lower_bounds_")
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            if col in self.lower_bounds_:
                X_df[col] = X_df[col].clip(
                    lower=self.lower_bounds_[col], upper=self.upper_bounds_[col]
                )
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        return self.feature_names_in_


class CyclicalDateTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to extract cyclical features from datetime columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            # Convert to datetime if it's not
            dt_col = pd.to_datetime(X[col], errors="coerce")

            # Extract month, day of week, hour
            month = dt_col.dt.month.fillna(1)
            dayofweek = dt_col.dt.dayofweek.fillna(0)
            hour = dt_col.dt.hour.fillna(0)

            # Apply sine/cosine
            out[f"{col}_month_sin"] = np.sin(2 * np.pi * month / 12)
            out[f"{col}_month_cos"] = np.cos(2 * np.pi * month / 12)
            out[f"{col}_dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
            out[f"{col}_dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)
            out[f"{col}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            out[f"{col}_hour_cos"] = np.cos(2 * np.pi * hour / 24)

        return out


class ReshapingTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Wrapper to handle 2D array inputs for TfidfVectorizer which expects 1D."""

    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            X = X[:, 0]
        self.vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            X = X[:, 0]
        return self.vectorizer.transform(
            X
        ).toarray()  # convert sparse to dense for downstream compatibility


from sklearn.impute import SimpleImputer


def get_imputer(strategy: str):
    if strategy == "median":
        return SimpleImputer(strategy="median")
    elif strategy == "mode":
        return SimpleImputer(strategy="most_frequent")
    elif strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value="")
    elif strategy == "passthrough":
        return FunctionTransformer(func=None, validate=False)
    else:
        return "drop"


def get_transformer(strategy: str):
    registry = {
        "standard": Pipeline(
            [
                ("winsorize", WinsorizerTransformer(lower=0.01, upper=0.99)),
                ("scale", StandardScaler()),
            ]
        ),
        "log": FunctionTransformer(func=np.log1p, validate=False),
        "power": PowerTransformer(),
        "ohe": OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=10,
            min_frequency=0.01,
        ),
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "tfidf": ReshapingTfidfVectorizer(max_features=100, stop_words="english"),
        "cyclical": CyclicalDateTransformer(),
        "passthrough": FunctionTransformer(func=None, validate=False),
        "drop": "drop",
    }
    return registry.get(strategy, "drop")
