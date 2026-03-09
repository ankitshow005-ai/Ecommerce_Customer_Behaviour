import os
import json
import pickle
import logging
import yaml
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# LOGGING
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("decision_tree")
logger.setLevel(logging.INFO)

if not logger.handlers:

    ch = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(LOG_DIR, "dt.log"))

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

def load_dependency_split(split_dir):

    X_train = pd.read_csv(os.path.join(split_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(split_dir, "X_test.csv"))

    y_train = pd.read_csv(os.path.join(split_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(split_dir, "y_test.csv")).squeeze()

    logger.info("Loaded dataset")

    return X_train, X_test, y_train, y_test


# ============================================================
# PARAMS
# ============================================================

def load_dt_config():

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    return params["decision_tree"]


# ============================================================
# TRAINING
# ============================================================

def train_decision_tree(X_train, y_train, config):

    tuning = config.get("tuning", False)

    if tuning:

        logger.info("Running GridSearch for Decision Tree")

        model = DecisionTreeClassifier(random_state=42)

        grid = GridSearchCV(
            model,
            param_grid=config["param_grid"],
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        model = grid.best_estimator_

        logger.info(f"Best params: {grid.best_params_}")

    else:

        logger.info("Training baseline Decision Tree")

        model = DecisionTreeClassifier(
            **config["params"],
            random_state=42
        )

        model.fit(X_train, y_train)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
    }

    logger.info(metrics)

    return metrics


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(model, metrics, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


# ============================================================
# MAIN
# ============================================================

def main():

    SPLIT_DIR = "data_processed/dependency_split"

    config = load_dt_config()

    variant = "tuned" if config.get("tuning") else "baseline"

    OUTPUT_DIR = f"models/decision_tree/{variant}"

    X_train, X_test, y_train, y_test = load_dependency_split(SPLIT_DIR)

    model = train_decision_tree(X_train, y_train, config)

    metrics = evaluate_model(model, X_test, y_test)

    save_artifacts(model, metrics, OUTPUT_DIR)


if __name__ == "__main__":
    main()