import os
import json
import pickle
import logging
import yaml
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("random_forest")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "rf.log"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ============================================================
# DATA LOADING
# ============================================================

def load_dependency_split(split_dir: str):
    """
    Load train-test split data prepared earlier.
    """
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
    """
    Load Random Forest configuration from params.yaml.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["random_forest"]

# ============================================================
# MODEL TRAINING
# ============================================================

def train_random_forest(X_train, y_train, config: dict):
    """
    Train Random Forest model.

    If tuning = False:
        → Baseline Random Forest

    If tuning = True:
        → GridSearchCV on structural params
        → Final model trained with fixed n_estimators
    """

    # ---------------- BASELINE RF ----------------
    if not config.get("tuning", False):
        logger.info("Training BASELINE Random Forest")

        model = RandomForestClassifier(
            **config["baseline_params"],
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    # ---------------- TUNED RF ----------------
    logger.info("Training TUNED Random Forest")

    base_model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=config["param_grid"],
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    logger.info(f"Best structural params from GridSearch: {best_params}")

    final_model = RandomForestClassifier(
        **best_params,
        **config["final_params"],   # n_estimators = 250
        random_state=42
    )

    final_model.fit(X_train, y_train)
    return final_model

# ============================================================
# THRESHOLD SWEEP (OPTIONAL)
# ============================================================

def threshold_sweep(model, X_test, y_test, sweep_cfg, output_dir):
    """
    Perform threshold sweep analysis and save results.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(
        sweep_cfg["start"],
        sweep_cfg["end"] + 1e-6,
        sweep_cfg["step"]
    )

    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        results.append({
            "threshold": round(float(t), 2),
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "recall": round(recall_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4)
        })

    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(output_dir, "threshold_analysis.csv"),
        index=False
    )

    logger.info("Threshold sweep completed")

# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, threshold):
    """
    Evaluate model using specified threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
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

def save_artifacts(model, metrics, output_dir):
    """
    Save trained model and metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Model and metrics saved successfully")

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    SPLIT_DIR = "data_processed/dependency_split"

    config = load_rf_config()
    variant = "tuned" if config.get("tuning", False) else "baseline"
    OUTPUT_DIR = f"models/random_forest/{variant}"

    # Create output directory early
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info(f"Starting Random Forest pipeline ({variant})")

    X_train, X_test, y_train, y_test = load_dependency_split(SPLIT_DIR)

    model = train_random_forest(X_train, y_train, config)

    # ---------------- OPTIONAL THRESHOLD LOGIC ----------------
    threshold_cfg = config.get("threshold", {})
    threshold_enabled = threshold_cfg.get("enabled", False)

    if threshold_enabled:
        logger.info("Threshold optimization ENABLED")

        threshold_sweep(
            model,
            X_test,
            y_test,
            threshold_cfg["sweep"],
            OUTPUT_DIR
        )

        final_threshold = threshold_cfg.get("value", 0.5)
    else:
        logger.info("Threshold optimization DISABLED")
        final_threshold = 0.5

    metrics = evaluate_model(model, X_test, y_test, final_threshold)

    save_artifacts(model, metrics, OUTPUT_DIR)

    logger.info("Random Forest pipeline completed successfully")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()