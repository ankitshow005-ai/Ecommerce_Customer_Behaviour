import os
import json
import pickle
import logging
import yaml
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.reporting import generate_reports


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("random_forest")
logger.setLevel(logging.INFO)

if not logger.handlers:

    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(LOG_DIR, "rf.log"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def load_dependency_split(split_dir):

    X_train = pd.read_csv(os.path.join(split_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(split_dir, "X_test.csv"))

    y_train = pd.read_csv(os.path.join(split_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(split_dir, "y_test.csv")).squeeze()

    logger.info("Loaded dependency split data")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def load_config():

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    return params


def train_random_forest(X_train, y_train, config):

    logger.info("Training Random Forest with tuned parameters")

    model = RandomForestClassifier(
        **config["random_forest"]["final_params"],
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, threshold):

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {

        "threshold": threshold,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),

        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn)
    }

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics


def save_artifacts(model, metrics, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Artifacts saved successfully")


def main():

    SPLIT_DIR = "data_processed/dependency_split"
    OUTPUT_DIR = "models/rf_tuned"

    config = load_config()

    threshold = config["random_forest"]["threshold"]["value"]

    logger.info("Starting Random Forest pipeline")

    X_train, X_test, y_train, y_test = load_dependency_split(SPLIT_DIR)

    model = train_random_forest(X_train, y_train, config)

    metrics = evaluate_model(model, X_test, y_test, threshold)

    generate_reports(
        model,
        X_test,
        y_test,
        threshold,
        X_train.columns.tolist()
    )

    save_artifacts(model, metrics, OUTPUT_DIR)

    logger.info("Random Forest pipeline completed successfully")


if __name__ == "__main__":
    main()