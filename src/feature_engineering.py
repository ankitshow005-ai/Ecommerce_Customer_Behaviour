import pandas as pd
import os
import logging

# ============================================================
# LOGGING SETUP
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "feature_engineering.log")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================

def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement-based features capturing user activity levels.
    """
    engagement_cols = [
        "Login_Frequency",
        "Session_Duration_Avg",
        "Pages_Per_Session",
        "Email_Open_Rate",
        "Mobile_App_Usage",
        "Social_Media_Engagement_Score",
    ]

    df["engagement_score"] = df[engagement_cols].mean(axis=1)

    threshold = df["engagement_score"].quantile(0.25)
    df["low_engagement_flag"] = (df["engagement_score"] <= threshold).astype(int)

    logger.info("Engagement features created")
    return df


def create_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create recency and inactivity-related features.
    """
    threshold = df["Days_Since_Last_Purchase"].quantile(0.75)
    df["inactive_flag"] = (df["Days_Since_Last_Purchase"] >= threshold).astype(int)

    logger.info("Recency features created")
    return df


def create_engagement_recency_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction feature between engagement and recency.
    """
    df["engagement_recency_risk"] = (
        df["low_engagement_flag"] * df["inactive_flag"]
    )

    logger.info("Engagement–recency interaction feature created")
    return df


def create_friction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features capturing user friction and dissatisfaction.
    """
    friction_cols = [
        "Cart_Abandonment_Rate",
        "Customer_Service_Calls",
    ]

    df["friction_score"] = df[friction_cols].mean(axis=1)

    threshold = df["friction_score"].quantile(0.75)
    df["high_friction_flag"] = (df["friction_score"] >= threshold).astype(int)

    logger.info("Friction features created")
    return df


def create_loyalty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create loyalty and commitment-related features.
    """
    loyalty_cols = [
        "Total_Purchases",
        "Product_Reviews_Written",
        "Wishlist_Items",
    ]

    df["loyalty_score"] = df[loyalty_cols].mean(axis=1)

    threshold = df["loyalty_score"].quantile(0.25)
    df["low_loyalty_flag"] = (df["loyalty_score"] <= threshold).astype(int)

    logger.info("Loyalty features created")
    return df


def save_featured_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save dataset with engineered features.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Feature engineered data saved to {output_path}")

# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def main():
    """
    Feature engineering pipeline.

    Flow:
    preprocessed.csv
      → engagement features
      → recency features
      → interaction features
      → friction features
      → loyalty features
      → features.csv
    """

    INPUT_PATH = "data_processed/preprocessed.csv"
    OUTPUT_PATH = "data_processed/features.csv"

    logger.info("Starting feature engineering stage")

    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded preprocessed data | Shape: {df.shape}")

    df = create_engagement_features(df)
    df = create_recency_features(df)
    df = create_engagement_recency_interaction(df)
    df = create_friction_features(df)
    df = create_loyalty_features(df)

    save_featured_data(df, OUTPUT_PATH)

    logger.info("Feature engineering stage completed successfully")

# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()