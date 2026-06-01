import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/processed/processed_data.csv")

assert "is_high_risk" in df.columns, "Target column missing!"

X = df.drop(columns=["is_high_risk", "CustomerId"])
y = df["is_high_risk"]


# ----------------------------
# SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ----------------------------
# MODELS + PARAM GRID
# ----------------------------
models = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1, 10]}
    ),
    "random_forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None]
        }
    )
}


# ----------------------------
# MLFLOW SETUP
# ----------------------------
mlflow.set_experiment("credit_risk_modeling")

best_model = None
best_f1 = 0
best_name = ""


# ----------------------------
# TRAINING LOOP
# ----------------------------
for name, (model, params) in models.items():

    print(f"Training {name}...")

    grid = GridSearchCV(
        model,
        params,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_

    preds = best_estimator.predict(X_test)
    probs = best_estimator.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)
    }

    with mlflow.start_run(run_name=name):

        mlflow.log_param("model_type", name)
        mlflow.log_params(grid.best_params_)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(best_estimator, "model")

    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_model = best_estimator
        best_name = name


print(f"\nBEST MODEL: {best_name} | F1: {best_f1}")