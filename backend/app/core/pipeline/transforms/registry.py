import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class EmailDomainExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X).astype(str).iloc[:, 0].str.split('@').str[-1].values.reshape(-1, 1)

class CyclicalDateTimeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Simplified cyclical for demo: extract month sin/cos
        dt = pd.to_datetime(pd.Series(X.ravel()))
        month_sin = np.sin(2 * np.pi * dt.dt.month / 12)
        month_cos = np.cos(2 * np.pi * dt.dt.month / 12)
        return np.column_stack([month_sin, month_cos])

TRANSFORM_REGISTRY = {
    "log": FunctionTransformer(np.log1p),
    "standard": StandardScaler(),
    "power": PowerTransformer(method='yeo-johnson'),
    "ohe": OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    "tfidf": TfidfVectorizer(max_features=100),
    "domain_extract": EmailDomainExtractor(),
    "cyclical": CyclicalDateTimeTransformer(),
    "passthrough": FunctionTransformer(None), # Identity
}
