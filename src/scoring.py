import pandas as pd

def combine_scores(df: pd.DataFrame, rule_col: str = "rule_alert", ml_col: str = "ml_alert") -> pd.DataFrame:
    """
    Combine rule and ML alerts into a single risk flag.
    A transaction is flagged if either the rule engine or the ML model raises an alert.
    """
    df = df.copy()
    df["flagged"] = df[rule_col] | df[ml_col]
    df["risk_level"] = "LOW"
    df.loc[df["rule_alert"] & ~df["ml_alert"], "risk_level"] = "MEDIUM"
    df.loc[df["ml_alert"] & ~df["rule_alert"], "risk_level"] = "MEDIUM"
    df.loc[df["rule_alert"] & df["ml_alert"], "risk_level"] = "HIGH"
    return df

def get_alert_summary(df: pd.DataFrame) -> dict:
    """Return high-level alert counts for dashboard display."""
    return {
        "total_transactions": len(df),
        "rule_alerts": int(df["rule_alert"].sum()),
        "ml_alerts": int(df["ml_alert"].sum()),
        "flagged_total": int(df["flagged"].sum()),
        "high_risk": int((df["risk_level"] == "HIGH").sum()),
        "medium_risk": int((df["risk_level"] == "MEDIUM").sum()),
        "low_risk": int((df["risk_level"] == "LOW").sum()),
    }