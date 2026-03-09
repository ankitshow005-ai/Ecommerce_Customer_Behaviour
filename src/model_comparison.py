import os
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path):

    with open(path) as f:
        return json.load(f)


def main():

    os.makedirs("reports", exist_ok=True)

    models = {
        "Decision Tree": "models/decision_tree/baseline/metrics.json",
        "Random Forest": "models/rf_tuned/metrics.json",
        "RF + Feature Selection": "models/rf_feature_selection/metrics.json"
    }

    rows = []

    for name, path in models.items():

        if os.path.exists(path):

            metrics = load_metrics(path)

            rows.append({
                "Model": name,
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1 Score": metrics.get("f1_score"),
                "ROC AUC": metrics.get("roc_auc")
            })

    df = pd.DataFrame(rows)

    # -------- SAVE TABLE --------

    df.to_csv("reports/model_comparison_table.csv", index=False)

    print("\nModel Comparison Table:\n")
    print(df)

    # -------- ACCURACY PLOT --------

    plt.figure(figsize=(8,5))
    plt.bar(df["Model"], df["Accuracy"])

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("reports/model_comparison_accuracy.png")
    plt.close()

    # -------- F1 SCORE PLOT --------

    plt.figure(figsize=(8,5))
    plt.bar(df["Model"], df["F1 Score"])

    plt.title("Model F1 Score Comparison")
    plt.ylabel("F1 Score")

    plt.tight_layout()
    plt.savefig("reports/model_comparison_f1.png")
    plt.close()


if __name__ == "__main__":
    main()