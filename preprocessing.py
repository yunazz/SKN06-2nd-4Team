import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# 이상치 처리
class OutlierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, whis=1.5):
        self.whis = whis

    def fit(self, X, y=None):
        q1 = np.nanquantile(X, q=0.25)
        q3 = np.nanquantile(X, q=0.75)
        IQR = q3 - q1
        self.lower_bound = q1 - IQR * self.whis
        self.upper_bound = q3 + IQR * self.whis
        return self

    def transform(self, X, y=None):
        X_transformed = np.where(X < self.lower_bound, self.lower_bound, X)
        X_transformed = np.where(
            X_transformed > self.upper_bound, self.upper_bound, X_transformed
        )
        return X_transformed


# 결측치 처리
class ProportionalImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.proportions = {}

    def fit(self, X, y=None):
        for column in X.columns:
            counts = X[column].value_counts(normalize=True, dropna=True)
            self.proportions[column] = counts
        return self

    def transform(self, X):
        X = X.copy()
        for column, probs in self.proportions.items():
            # 결측치 위치 찾기
            missing_mask = X[column].isna()
            if missing_mask.any():
                # 비율에 따라 랜덤하게 값 채우기
                X.loc[missing_mask, column] = np.random.choice(probs.index, size=missing_mask.sum(), p=probs.values)
        return X


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X).reshape(-1, 1)  

# 순서 인코딩
class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categories=[]):
        print(categories)
        self.encoder = OrdinalEncoder(categories=categories)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        return self.encoder.transform(X)
