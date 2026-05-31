import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
            Most_Common_Product=("ProductCategory", lambda x: x.mode()[0] if not x.mode().empty else "unknown"),
            Most_Common_Channel=("ChannelId", lambda x: x.mode()[0] if not x.mode().empty else "unknown"),
            Most_Common_Provider=("ProviderId", lambda x: x.mode()[0] if not x.mode().empty else "unknown")
        ).reset_index()

        # ----------------------------
        # MERGE ALL FEATURES
        # ----------------------------
        final_df = agg.merge(cat, on="CustomerId", how="left")

        return final_df