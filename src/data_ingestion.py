import pandas as pd
import os
import logging

# ============================================================
# LOGGING SETUP
# ============================================================

# Create logs directory if it does not exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a logger specific to data ingestion
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if script is run multiple times
if not logger.handlers:

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "data_ingestion.log")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ============================================================
# DATA INGESTION FUNCTIONS
# ============================================================

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data from raw_data_location.

    This function:
    - Only reads data
    - Does NOT modify data
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Raw data loaded from {file_path}")
        logger.info(f"Data shape: {df.shape}")
        return df

    except Exception as e:
        logger.error("Error while loading raw data", exc_info=True)
        raise


def save_ingested_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save ingested data to data_processed/raw directory.

    This directory is owned by the data_ingestion stage.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Ingested data saved to {output_path}")

    except Exception as e:
        logger.error("Error while saving ingested data", exc_info=True)
        raise

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution for data ingestion stage.

    Pipeline flow:
    raw_data_location
        ↓
    data_ingestion
        ↓
    data_processed/raw/ingested.csv
    """

    # Input: raw data (pipeline dependency)
    RAW_DATA_PATH = "raw_data_location/ecommerce_customer_churn_dataset.csv"

    # Output: ingestion stage output directory (DVC output)
    OUTPUT_DATA_PATH = "data_processed/raw/ingested.csv"

    logger.info("Starting data ingestion stage")

    # Step 1: Load raw data
    df = load_raw_data(RAW_DATA_PATH)

    # Step 2: Save ingested data
    save_ingested_data(df, OUTPUT_DATA_PATH)

    logger.info("Data ingestion stage completed successfully")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()