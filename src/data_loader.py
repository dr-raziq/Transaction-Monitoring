import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "PS_20174392719_1491204439457_log.csv"

def load_transactions(path: Path = DATA_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Load PaySim CSV, rename columns for clarity, and derive basic features."""
    df = pd.read_csv(path, nrows=nrows)
    df = df.rename(columns={
        "oldbalanceOrg": "oldbalanceOrig",
        "newbalanceOrg": "newbalanceOrig",
        "oldbalanceDest": "oldbalanceDest",
        "newbalanceDest": "newbalanceDest",
    })
    df["orig_balance_delta"] = df["oldbalanceOrig"] - df["newbalanceOrig"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return df

def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature matrix used by the anomaly model."""
    feature_cols = [
        "amount",
        "oldbalanceOrig",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "orig_balance_delta",
        "dest_balance_delta",
    ]
    return df[feature_cols].copy()