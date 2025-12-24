import pandas as pd
import logging
import os

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_validation")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "data_validation.log"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load ingested data for validation."""
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded for validation | Shape: {df.shape}")
    return df


def get_missing_value_percentage(df: pd.DataFrame) -> pd.Series:
    """Calculate missing value percentage for each column."""
    return round(df.isnull().mean() * 100, 2)


def check_row_duplicates(df: pd.DataFrame) -> bool:
    """Check if duplicate rows exist."""
    return df.duplicated().any()


def check_column_duplicates(df: pd.DataFrame) -> bool:
    """Check if duplicate columns exist."""
    return df.shape[1] != df.T.drop_duplicates().T.shape[1]


def get_categorical_and_numerical_columns(df: pd.DataFrame):
    """Separate categorical and numerical columns."""
    cat_cols = [col for col in df.columns if df[col].dtype == "O"]
    num_cols = [col for col in df.columns if df[col].dtype != "O"]
    return cat_cols, num_cols


def get_single_unique_columns(df: pd.DataFrame):
    """Identify columns with a single unique value."""
    return [col for col in df.columns if df[col].nunique() == 1]


def get_high_missing_columns(df: pd.DataFrame, threshold: float = 30.0):
    """Identify columns with missing values greater than threshold."""
    return [
        col for col in df.columns
        if round(df[col].isnull().mean() * 100, 2) > threshold
    ]


def get_columns_to_remove(
    single_unique_cols,
    high_missing_cols,
    non_important_cols=None
):
    """Create consolidated removal list."""
    if non_important_cols is None:
        non_important_cols = []

    return list(set(single_unique_cols + high_missing_cols + non_important_cols))


# ============================================================
# MAIN VALIDATION LOGIC
# ============================================================

def main():
    DATA_PATH = "data_processed/ingested.csv"

    logger.info("Starting data validation stage")

    df = load_data(DATA_PATH)

    # Data volume
    logger.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Missing values
    missing_pct = get_missing_value_percentage(df)
    logger.info(f"Missing value percentage:\n{missing_pct}")

    # Duplicate checks
    if check_row_duplicates(df):
        logger.warning("Duplicate rows found")
    else:
        logger.info("No duplicate rows found")

    if check_column_duplicates(df):
        logger.warning("Duplicate columns found")
    else:
        logger.info("No duplicate columns found")

    # Categorical & Numerical columns
    cat_cols, num_cols = get_categorical_and_numerical_columns(df)
    logger.info(f"Categorical columns: {cat_cols}")
    logger.info(f"Numerical columns: {num_cols}")

    # Single unique columns
    single_unique_cols = get_single_unique_columns(df)
    logger.info(f"Single unique columns: {single_unique_cols}")

    # High missing columns
    high_missing_cols = get_high_missing_columns(df)
    logger.info(f"Columns with >30% missing values: {high_missing_cols}")

    # Non-important columns (empty for now, can be expanded later)
    rmv_list = []

    # Final removal list
    removal_list = get_columns_to_remove(
        single_unique_cols,
        high_missing_cols,
        rmv_list
    )

    logger.info(f"Total columns to remove: {len(removal_list)}")
    logger.info(f"Columns marked for removal: {removal_list}")

    logger.info("Data validation stage completed")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()