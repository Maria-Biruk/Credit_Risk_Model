import pandas as pd
from src.data_processing import create_rfm_features


def test_rfm_contains_columns():

    data = pd.DataFrame({
        "CustomerId": [1, 2, 3],
        "TransactionId": [101, 102, 103],
        "Amount": [100, 200, 300],
        "TransactionStartTime": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03"
        ]
    })

    result = create_rfm_features(data)

    expected_cols = [
        "CustomerId",
        "Recency",
        "Frequency",
        "Monetary"
    ]

    for col in expected_cols:
        assert col in result.columns


def test_rfm_not_empty():

    data = pd.DataFrame({
        "CustomerId": [1, 2, 3],
        "TransactionId": [101, 102, 103],
        "Amount": [100, 200, 300],
        "TransactionStartTime": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03"
        ]
    })

    result = create_rfm_features(data)

    assert len(result) > 0
