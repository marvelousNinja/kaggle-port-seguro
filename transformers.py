from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict, cross_val_score
import pandas as pd

class ModelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column, value, model_factory, debug=False):
        self.column = column
        self.value = value
        self.model_factory = model_factory
        self.debug = debug
    def fit(self, X, y=None):
        self.model = self.model_factory()
        X = X[X[self.column] != self.value]
        self.model.fit(X.drop(self.column, axis=1), X[self.column])
        if self.debug:
            scores = cross_val_score(self.model, X.drop(self.column, axis=1), X[self.column], cv=3)
            print('Col:', self.column, 'Col Mean:', X[self.column].astype('float').mean(), 'Mean:', scores.mean(), 'Std:', scores.std())
        return self
    def transform(self, X, y=None):
        X = X.copy()
        targets = X[X[self.column] == self.value]
        predictions = self.model.predict(targets.drop(self.column, axis=1))
        X.loc[X[self.column] == self.value, self.column] = predictions
        return X

class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns
    def fit(self, X, y=None):
        self.category_mapping = dict()

        if not self.columns:
            self.columns = list(X.columns)

        for column in self.columns:
            self.category_mapping[column] = X[column].astype('category').cat.categories.values
        return self
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column] = X[column].astype('object').astype('category', categories=self.category_mapping[column])
        return pd.get_dummies(X, columns=self.columns, drop_first=True)

## TODO AS: Seems to overfit badly
import numpy as np
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, target, min_samples_leaf, noise_level, smoothing):
        self.columns = columns
        self.target = target
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.smoothing = smoothing
    def fit(self, X, y=None):
        self.mapping = dict()
        self.global_mean = X[self.target].mean()
        for column in self.columns:
            self.mapping[column] = dict()
            averages = X.groupby([column])[self.target].agg(['mean', 'count'])
            smoothing_coefs = 1 / (1 + np.exp(-(averages['count'] - self.min_samples_leaf) / self.smoothing))
            print(1 - smoothing_coefs)
            values = self.global_mean * (1 - smoothing_coefs) + averages['mean'] * smoothing_coefs
            self.mapping[column] = dict(values)
        return self
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column + '_' + self.target] = X[column].map(self.mapping[column], na_action=self.global_mean)
            noise = 1 + self.noise_level * np.random.randn(len(X))
            X[column + '_' + self.target] = X[column] * noise
        return X

class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        self.mapping = dict()
        for column in self.columns:
            self.mapping[column] = dict()
            for value in X[column].unique():
                self.mapping[column][value] = len(X[X[column] == value])
        return self
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column] = X[column].map(self.mapping[column], na_action=1)
        return X

class CountRankEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        self.mapping = dict()
        for column in self.columns:
            self.mapping[column] = dict()
            counts = X[column].value_counts()
            for i, value in enumerate(counts.index):
                self.mapping[column][value] = i
        return self
    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            X[column] = X[column].map(self.mapping[column], na_action=max(self.mapping[column].values()) + 1)
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(self.columns, axis=1)

class ColumnBinner(BaseEstimator, TransformerMixin):
    def __init__(self, column, bins=7, labels=False):
        self.column = column
        self.bins = bins
        self.labels = labels
    def fit(self, X, y=None):
        self.fitted_bins = pd.cut(X[self.column], bins=self.bins, right=False, labels=self.labels, retbins=True)[1]
        return self
    def transform(self, X):
        X = X.copy()
        X[self.column] = pd.cut(X[self.column], bins=self.fitted_bins, right=False, labels=self.labels)
        return X

class ColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, func):
        self.column = column
        self.func = func
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X[self.column] = self.func(X)
        return X

import numpy as np
class ColumnClipper(BaseEstimator, TransformerMixin):
    def __init__(self, column, lower_percentile, upper_percentile):
        self.column = column
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
    def fit(self, X, y=None):
        self.lower, self.upper = np.percentile(X[self.column], [self.lower_percentile, self.upper_percentile])
        return self
    def transform(self, X):
        X = X.copy()
        X[self.column] = np.clip(X[self.column], self.lower, self.upper)
        return X
