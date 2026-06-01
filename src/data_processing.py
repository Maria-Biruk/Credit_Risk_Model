import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class CreditFeatureEngineer:
    def transform(self, df):
        df = df.copy()

        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

        agg = df.groupby("CustomerId").agg(
            Total_Transaction_Amount=("Amount", "sum"),
            Avg_Transaction_Amount=("Amount", "mean"),
            Transaction_Count=("TransactionId", "count"),
            Std_Transaction_Amount=("Amount", "std"),
            Avg_Transaction_Hour=("TransactionStartTime", lambda x: x.dt.hour.mean()),
            Most_Common_Month=("TransactionStartTime", lambda x: x.dt.month.mode()[0])
        ).reset_index()

        agg["Std_Transaction_Amount"] = agg["Std_Transaction_Amount"].fillna(0)

        return agg


def create_rfm_features(df):

    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = df["TransactionStartTime"].max()

    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled)

    cluster_summary = rfm.groupby("cluster")[["Frequency", "Monetary"]].mean()

    high_risk_cluster = cluster_summary["Frequency"].idxmin()

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm