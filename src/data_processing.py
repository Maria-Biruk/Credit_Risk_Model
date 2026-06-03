import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# =========================
# DATE FEATURES
# =========================
class DateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        df["transaction_hour"] = df["TransactionStartTime"].dt.hour
        df["transaction_day"] = df["TransactionStartTime"].dt.day
        df["transaction_month"] = df["TransactionStartTime"].dt.month
        df["transaction_year"] = df["TransactionStartTime"].dt.year

        return df


# =========================
# AGGREGATION
# =========================
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        agg = df.groupby("CustomerId").agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
            max_transaction_amount=("Amount", "max"),
            min_transaction_amount=("Amount", "min")
        ).reset_index()

        agg["std_transaction_amount"] = agg["std_transaction_amount"].fillna(0)

        return agg


# =========================
# PREPROCESSOR
# =========================
def build_preprocessor():
    numeric_features = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "max_transaction_amount",
        "min_transaction_amount"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features)
    ])


# =========================
# PIPELINE
# =========================
def build_pipeline():
    return Pipeline(steps=[
        ("date_features", DateFeatures()),
        ("aggregation", AggregateFeatures()),
        ("preprocessor", build_preprocessor())
    ])
