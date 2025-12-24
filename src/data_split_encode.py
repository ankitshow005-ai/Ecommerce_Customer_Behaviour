import pandas as pd
import os
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_split_encode")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "data_split_encode.log")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_feature_data(file_path: str) -> pd.DataFrame:
    """
    Load final feature engineered dataset.
    """
    df = pd.read_csv(file_path)
    logger.info(f"Loaded feature data | Shape: {df.shape}")
    return df


def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Separate features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def perform_train_test_split(
    X, y, test_size=0.2, random_state=42
):
    """
    Perform train–test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    logger.info("Train–test split completed")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def encode_categorical_features(X_train, X_test):
    """
    Apply Label Encoding to categorical columns.
    Encoder is fit ONLY on training data.
    """
    cat_cols = [col for col in X_train.columns if X_train[col].dtype == "O"]
    encoders = {}

    logger.info(f"Categorical columns detected: {cat_cols}")

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    logger.info("Label encoding applied successfully")
    return X_train, X_test, encoders


def save_dependency_split(
    X_train, X_test, y_train, y_test, encoders, output_dir
):
    """
    Save train/test splits and encoders.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    with open(os.path.join(output_dir, "label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    logger.info("Dependency split data saved successfully")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Pipeline:
    features.csv
      → train-test split
      → label encoding
      → save dependency split
    """

    INPUT_PATH = "data_processed/features.csv"
    OUTPUT_DIR = "data_processed/dependency_split"
    TARGET_COLUMN = "Churned"

    logger.info("Starting train-test split & encoding stage")

    df = load_feature_data(INPUT_PATH)
    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = perform_train_test_split(X, y)
    X_train, X_test, encoders = encode_categorical_features(X_train, X_test)

    save_dependency_split(
        X_train, X_test, y_train, y_test, encoders, OUTPUT_DIR
    )

    logger.info("Split & encoding stage completed successfully")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()