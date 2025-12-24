import os
import json
import pickle
import logging
import yaml
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    return X_train, X_test, y_train, y_test

# ============================================================
# PARAMS LOADING
# ============================================================

def load_rf_fs_config():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["random_forest_feature_selection"]

# ============================================================
# FEATURE IMPORTANCE + SELECTION
# ============================================================

def compute_feature_importance(model, feature_names):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "column_name": feature_names,
        "mean_impurity": importances
    })

    df["percentage_impurity"] = round(df["mean_impurity"] * 100, 3)
    df = df.sort_values(by="percentage_impurity", ascending=False)
    df["cum_impurity"] = df["percentage_impurity"].cumsum()

    df = df.reset_index(drop=True)
    return df

def select_features(importance_df, threshold):
    selected_features = list(
        importance_df[importance_df["cum_impurity"] <= threshold]["column_name"]
    )

    logger.info(f"Selected {len(selected_features)} features using {threshold}% threshold")
    return selected_features

# ============================================================
# MODEL TRAINING
# ============================================================

def train_random_forest(X_train, y_train, params):
    model = RandomForestClassifier(
        **params,
        random_state=42
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
        "f1_score": round(f1_score(y_test, y_pred), 4)
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(model, metrics, importance_df, selected_features, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save feature importance
    importance_df.to_csv(
        os.path.join(output_dir, "feature_importance.csv"),
        index=False
    )

    # Save selected features
    with open(os.path.join(output_dir, "selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=4)

    logger.info("Artifacts saved successfully")

# ============================================================
# MAIN
# ============================================================

def main():
    SPLIT_DIR = "data_processed/dependency_split"
    OUTPUT_DIR = "models/random_forest/tuned_feature_selection"

    logger.info("Starting Random Forest Feature Selection pipeline")

    X_train, X_test, y_train, y_test = load_dependency_split(SPLIT_DIR)
    config = load_rf_fs_config()

    # Step 1: Train base RF model
    base_model = train_random_forest(
        X_train,
        y_train,
        config["base_model_params"]
    )

    # Step 2: Compute feature importance
    importance_df = compute_feature_importance(
        base_model,
        X_train.columns.tolist()
    )

    # Step 3: Select features
    selected_features = select_features(
        importance_df,
        config["cumulative_importance_threshold"]
    )

    # Step 4: Filter data
    X_train_fs = X_train[selected_features]
    X_test_fs = X_test[selected_features]

    # Step 5: Retrain RF on selected features
    final_model = train_random_forest(
        X_train_fs,
        y_train,
        config["final_model_params"]
    )

    # Step 6: Evaluate
    metrics = evaluate_model(final_model, X_test_fs, y_test)

    # Step 7: Save everything
    save_artifacts(
        final_model,
        metrics,
        importance_df,
        selected_features,
        OUTPUT_DIR
    )

    logger.info("Random Forest Feature Selection pipeline completed")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()