import pandas as pd
from typing import Dict, List

# Rule configuration – thresholds are illustrative and should be calibrated to the institution's risk appetite and the customer's expected behaviour.
DEFAULT_RULES = {
    "large_transfer": {"threshold": 100_000, "types": ["TRANSFER", "CASH_OUT"]},
    "high_frequency": {"max_transactions_per_hour": 5},
    "structuring": {"min_amount": 9_000, "max_amount": 9_999, "window_hours": 24},
    "rapid_drain": {"drain_ratio": 0.9, "max_minutes": 20},
}

def apply_rules(df: pd.DataFrame, rules: Dict = None) -> pd.DataFrame:
    if rules is None:
        rules = DEFAULT_RULES
    df = df.copy()
    df["alert_large_transfer"] = False
    df["alert_high_frequency"] = False
    df["alert_structuring"] = False
    df["alert_rapid_drain"] = False

    # Rule 1: Large transfer
    lt = rules["large_transfer"]
    mask_lt = (df["amount"] > lt["threshold"]) & (df["type"].isin(lt["types"]))
    df.loc[mask_lt, "alert_large_transfer"] = True

    # Rule 2: High frequency per originating account per hour
    freq = df.groupby(["nameOrig", "step"]).size().reset_index(name="count")
    freq = freq[freq["count"] > rules["high_frequency"]["max_transactions_per_hour"]]
    high_freq_keys = set(zip(freq["nameOrig"], freq["step"]))
    df["alert_high_frequency"] = df.apply(
        lambda r: (r["nameOrig"], r["step"]) in high_freq_keys, axis=1
    )

    # Rule 3: Structuring – multiple transactions just below reporting threshold
    st = rules["structuring"]
    struct_mask = (
        (df["amount"] >= st["min_amount"])
        & (df["amount"] <= st["max_amount"])
        & (df["type"].isin(["CASH_IN", "CASH_OUT", "TRANSFER"]))
    )
    struct_df = df[struct_mask].copy()
    if not struct_df.empty:
        struct_df["hour_bucket"] = struct_df["step"]
        counts = struct_df.groupby(["nameOrig", "hour_bucket"]).size().reset_index(name="cnt")
        suspicious = counts[counts["cnt"] >= 3]
        struct_keys = set(zip(suspicious["nameOrig"], suspicious["hour_bucket"]))
        df["alert_structuring"] = df.apply(
            lambda r: (r["nameOrig"], r["step"]) in struct_keys, axis=1
        )

    # Rule 4: Rapid drain – account receives funds then sends most out within 20 minutes
    rd = rules["rapid_drain"]
    incoming = df[df["type"].isin(["TRANSFER", "CASH_IN"])].copy()
    outgoing = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    for _, inc in incoming.iterrows():
        window = outgoing[
            (outgoing["nameOrig"] == inc["nameDest"])
            & (outgoing["step"] >= inc["step"])
            & (outgoing["step"] <= inc["step"] + 1)  # 1 step ≈ 1 hour
        ]
        if not window.empty:
            total_out = window["amount"].sum()
            if total_out >= rd["drain_ratio"] * inc["amount"]:
                df.loc[window.index, "alert_rapid_drain"] = True

    df["rule_alert"] = df[
        ["alert_large_transfer", "alert_high_frequency", "alert_structuring", "alert_rapid_drain"]
    ].any(axis=1)
    return df

def get_rule_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return count of alerts triggered by each rule."""
    return {
        "large_transfer": int(df["alert_large_transfer"].sum()),
        "high_frequency": int(df["alert_high_frequency"].sum()),
        "structuring": int(df["alert_structuring"].sum()),
        "rapid_drain": int(df["alert_rapid_drain"].sum()),
        "total_rule_alerts": int(df["rule_alert"].sum()),
    }