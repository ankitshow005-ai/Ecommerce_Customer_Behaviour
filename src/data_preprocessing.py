import pandas as pd
import os
import logging

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "data_preprocessing.log")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load ingested data."""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded ingested data | Shape: {df.shape}")
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows before any statistical operation."""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    logger.info(f"Removed {before - after} duplicate rows")
    return df


def remove_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """Remove columns decided during validation."""
    if columns_to_remove:
        logger.info(f"Removing columns: {columns_to_remove}")
        df = df.drop(columns=columns_to_remove, errors="ignore")
    else:
        logger.info("No columns marked for removal")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values using median for numerical columns."""
    num_cols = df.select_dtypes(include=["int", "float"]).columns

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            logger.info(f"Imputed missing values in '{col}' with median")

    return df


# ---------------- OUTLIER HANDLING ----------------

def cap_outliers_iqr(df: pd.DataFrame, col: str) -> None:
    """Cap outliers using IQR method."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = df[col].clip(lower, upper)


def handle_outliers(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Apply IQR capping to numerical columns except target."""
    num_cols = df.select_dtypes(include=["int", "float"]).columns
    num_cols = [col for col in num_cols if col != target_column]

    logger.info(f"Applying outlier handling on columns: {num_cols}")

    for col in num_cols:
        cap_outliers_iqr(df, col)

    return df


def save_preprocessed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Data preprocessing pipeline.

    Order:
    1. Load data
    2. Remove duplicate rows
    3. Remove unwanted columns
    4. Impute missing values
    5. Handle outliers
    6. Save cleaned data
    """

    INPUT_PATH = "data_processed/raw/ingested.csv"
    OUTPUT_PATH = "data_processed/cleaned/cleaned.csv"

    TARGET_COLUMN = "Churned"

    # To be filled later from validation / params.yaml
    COLUMNS_TO_REMOVE = []

    logger.info("Starting data preprocessing stage")

    df = load_data(INPUT_PATH)
    df = remove_duplicate_rows(df)
    df = remove_columns(df, COLUMNS_TO_REMOVE)
    df = impute_missing_values(df)
    df = handle_outliers(df, TARGET_COLUMN)

    save_preprocessed_data(df, OUTPUT_PATH)

    logger.info("Data preprocessing stage completed successfully")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()