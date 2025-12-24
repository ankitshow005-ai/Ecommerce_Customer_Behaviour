import os
import json
import pickle
import logging
import yaml
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("random_forest_feature_selection")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(LOG_DIR, "rf_fs.log"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# ============================================================
# DATA LOADING
# ============================================================

def load_dependency_split(split_dir: str):
    X_train = pd.read_csv(os.path.join(split_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(split_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(split_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(split_dir, "y_test.csv")).squeeze()

    logger.info("Loaded dependency split data")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

# ============================================================
# PARAMS LOADING
# ============================================================

def load_rf_config():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["random_forest"]

# ============================================================
# TRAIN BASE RF (FOR FEATURE IMPORTANCE)
# ============================================================

def train_base_rf(X_train, y_train, config):
    """
    Train RF using tuned structural params and final n_estimators.
    """
    logger.info("Training Random Forest for feature importance")

    model = RandomForestClassifier(
        **config["final_params"],
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

# ============================================================
# FEATURE IMPORTANCE LOGIC
# ============================================================

def extract_feature_importance(model, feature_names):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    df["importance_pct"] = df["importance"] * 100
    df["cumulative_importance"] = df["importance_pct"].cumsum()

    return df.reset_index(drop=True)


def select_features(df, threshold=90.0):
    selected = df[df["cumulative_importance"] <= threshold]["feature"].tolist()
    logger.info(f"Selected {len(selected)} features using {threshold}% threshold")
    return selected

# ============================================================
# FINAL MODEL TRAINING
# ============================================================

def train_final_model(X_train, y_train, config):
    model = RandomForestClassifier(
        **config["final_params"],
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    metrics,
    importance_df,
    selected_features,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    importance_df.to_csv(
        os.path.join(output_dir, "feature_importance.csv"),
        index=False
    )

    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=4)

    logger.info("All artifacts saved successfully")

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    SPLIT_DIR = "data_processed/dependency_split"
    OUTPUT_DIR = "models/random_forest/tuned_feature_selection"
    FEATURE_THRESHOLD = 90.0

    logger.info("Starting RF Feature Selection pipeline")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_dependency_split(SPLIT_DIR)
    config = load_rf_config()

    # Train RF for importance
    base_model = train_base_rf(X_train, y_train, config)

    # Feature importance
    importance_df = extract_feature_importance(
        base_model,
        X_train.columns.tolist()
    )

    selected_features = select_features(
        importance_df,
        FEATURE_THRESHOLD
    )

    # Filter data
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Train final model
    final_model = train_final_model(X_train_sel, y_train, config)

    metrics = evaluate_model(final_model, X_test_sel, y_test)

    save_artifacts(
        final_model,
        metrics,
        importance_df,
        selected_features,
        OUTPUT_DIR
    )

    logger.info("RF Feature Selection pipeline completed successfully")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()