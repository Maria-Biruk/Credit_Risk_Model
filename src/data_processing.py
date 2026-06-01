import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def get_categorical_features(self, df):
        return df[[
            "ProductCategory",
            "ChannelId",
            "ProviderId"
        ]]

    def transform(self, X):
        df = X.copy()

        # ----------------------------
        # BASIC CLEANING
        # ----------------------------
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

        # ----------------------------
        # TIME FEATURE ENGINEERING
        # ----------------------------
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        df["Transaction_Hour"] = df["TransactionStartTime"].dt.hour
        df["Transaction_Day"] = df["TransactionStartTime"].dt.day
        df["Transaction_Month"] = df["TransactionStartTime"].dt.month
        df["Transaction_Year"] = df["TransactionStartTime"].dt.year

        # ----------------------------
        # AGGREGATION (NUMERIC FEATURES)
        # ----------------------------
        agg = df.groupby("CustomerId").agg(
            Total_Transaction_Amount=("Amount", "sum"),
            Avg_Transaction_Amount=("Amount", "mean"),
            Transaction_Count=("TransactionId", "count"),
            Std_Transaction_Amount=("Amount", "std"),
            Avg_Transaction_Hour=("Transaction_Hour", "mean"),
            Most_Common_Month=("Transaction_Month", "median")
        ).reset_index()

        agg["Std_Transaction_Amount"] = agg["Std_Transaction_Amount"].fillna(0)

        # ----------------------------
        # CATEGORICAL FEATURES
        # ----------------------------
        cat = df.groupby("CustomerId").agg(
            Most_Common_Product=(
                "ProductCategory",
                lambda x: x.mode()[0] if not x.mode().empty else "unknown"
            ),
            Most_Common_Channel=(
                "ChannelId",
                lambda x: x.mode()[0] if not x.mode().empty else "unknown"
            ),
            Most_Common_Provider=(
                "ProviderId",
                lambda x: x.mode()[0] if not x.mode().empty else "unknown"
            )
        ).reset_index()

        # ----------------------------
        # MERGE ALL FEATURES
        # ----------------------------
        final_df = agg.merge(cat, on="CustomerId", how="left")

        return final_df


# ==========================================
# TASK 4: RFM FEATURE CREATION
# ==========================================

def create_rfm_features(df):

    data = df.copy()

    data["TransactionStartTime"] = pd.to_datetime(
        data["TransactionStartTime"]
    )

    snapshot_date = (
        data["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = data.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (snapshot_date - x.max()).days
        ),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    return rfm 
def add_rfm_and_target(df, engineered_df):
    rfm = create_rfm_features(df)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # identify high-risk cluster
    summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = summary["Recency"].idxmax()

    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    final_df = engineered_df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    return final_df