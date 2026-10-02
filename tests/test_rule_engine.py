import pandas as pd
from src.rule_engine import apply_rules, get_rule_summary

def test_large_transfer_rule():
    df = pd.DataFrame({
        "step": [1],
        "type": ["TRANSFER"],
        "amount": [150_000],
        "nameOrig": ["C1"],
        "oldbalanceOrig": [200_000],
        "newbalanceOrig": [50_000],
        "nameDest": ["C2"],
        "oldbalanceDest": [0],
        "newbalanceDest": [150_000],
    })
    result = apply_rules(df)
    assert result["alert_large_transfer"].iloc[0] == True
    assert result["rule_alert"].iloc[0] == True

def test_no_alert_for_small_amount():
    df = pd.DataFrame({
        "step": [1],
        "type": ["PAYMENT"],
        "amount": [500],
        "nameOrig": ["C1"],
        "oldbalanceOrig": [10_000],
        "newbalanceOrig": [9_500],
        "nameDest": ["M1"],
        "oldbalanceDest": [0],
        "newbalanceDest": [500],
    })
    result = apply_rules(df)
    assert result["rule_alert"].iloc[0] == False

def test_rule_summary_keys():
    df = pd.DataFrame({
        "step": [1, 2],
        "type": ["TRANSFER", "CASH_OUT"],
        "amount": [150_000, 50_000],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrig": [200_000, 100_000],
        "newbalanceOrig": [50_000, 50_000],
        "nameDest": ["C2", "C3"],
        "oldbalanceDest": [0, 0],
        "newbalanceDest": [150_000, 50_000],
    })
    result = apply_rules(df)
    summary = get_rule_summary(result)
    assert "large_transfer" in summary
    assert "total_rule_alerts" in summary