import os
import json
import pickle
import logging
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


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


def load_split(split_dir):

    X_train = pd.read_csv(os.path.join(split_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(split_dir, "X_test.csv"))

    y_train = pd.read_csv(os.path.join(split_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(split_dir, "y_test.csv")).squeeze()

    return X_train, X_test, y_train, y_test


def load_config():

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    return params


def train_rf(X_train, y_train, config):

    model = RandomForestClassifier(
        **config["random_forest"]["final_params"],
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def extract_importance(model, features):

    df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    df["importance_pct"] = df["importance"] * 100
    df["cumulative_importance"] = df["importance_pct"].cumsum()

    return df


def plot_feature_importance(df):

    os.makedirs("reports", exist_ok=True)

    top_features = df.head(15)

    plt.figure(figsize=(10,6))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.gca().invert_yaxis()

    plt.title("Random Forest Feature Importance")

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig("reports/feature_importance.png")

    plt.close()


def select_features(df, threshold):

    selected = df[df["cumulative_importance"] <= threshold]["feature"].tolist()

    logger.info(f"{len(selected)} features selected")

    return selected


def evaluate(model, X_test, y_test, threshold):

    y_prob = model.predict_proba(X_test)[:,1]
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
        "false_positives": int(fp)

    }

    return metrics


def save_artifacts(model, metrics, importance_df, features, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    importance_df.to_csv(
        os.path.join(out_dir, "feature_importance.csv"),
        index=False
    )

    with open(os.path.join(out_dir, "selected_features.json"), "w") as f:
        json.dump(features, f, indent=4)


def main():

    SPLIT_DIR = "data_processed/dependency_split"
    OUTPUT_DIR = "models/rf_feature_selection"

    config = load_config()

    threshold = config["random_forest"]["threshold"]["value"]
    feature_threshold = config["feature_selection"]["importance_threshold"]

    X_train, X_test, y_train, y_test = load_split(SPLIT_DIR)

    base_model = train_rf(X_train, y_train, config)

    importance_df = extract_importance(base_model, X_train.columns)

    # CREATE FEATURE IMPORTANCE PLOT
    plot_feature_importance(importance_df)

    selected = select_features(importance_df, feature_threshold)

    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]

    final_model = train_rf(X_train_sel, y_train, config)

    metrics = evaluate(final_model, X_test_sel, y_test, threshold)

    save_artifacts(final_model, metrics, importance_df, selected, OUTPUT_DIR)

    logger.info("RF Feature Selection completed")


if __name__ == "__main__":
    main()