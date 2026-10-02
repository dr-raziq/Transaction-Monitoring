import pandas as pd
from src.data_loader import get_feature_matrix
from src.anomaly_model import train_anomaly_model, score_anomalies

def test_anomaly_model_trains_and_scores():
    df = pd.DataFrame({
        "amount": [100, 200, 300, 1000, 5000, 10000],
        "oldbalanceOrig": [1000, 900, 800, 5000, 20000, 50000],
        "newbalanceOrig": [900, 700, 500, 4000, 15000, 40000],
        "oldbalanceDest": [0, 0, 0, 0, 0, 0],
        "newbalanceDest": [100, 200, 300, 1000, 5000, 10000],
        "orig_balance_delta": [100, 200, 300, 1000, 5000, 10000],
        "dest_balance_delta": [100, 200, 300, 1000, 5000, 10000],
    })
    features = get_feature_matrix(df)
    model, scaler = train_anomaly_model(features, contamination=0.2)
    scores = score_anomalies(model, scaler, features)
    assert "anomaly_score" in scores.columns
    assert "ml_alert" in scores.columns
    assert scores["ml_alert"].sum() > 0  # with 20% contamination some flags expected

def test_anomaly_scores_are_numeric():
    df = pd.DataFrame({
        "amount": [50, 100, 150],
        "oldbalanceOrig": [500, 450, 400],
        "newbalanceOrig": [450, 350, 250],
        "oldbalanceDest": [0, 0, 0],
        "newbalanceDest": [50, 100, 150],
        "orig_balance_delta": [50, 100, 150],
        "dest_balance_delta": [50, 100, 150],
    })
    features = get_feature_matrix(df)
    model, scaler = train_anomaly_model(features, contamination=0.1)
    scores = score_anomalies(model, scaler, features)
    assert scores["anomaly_score"].dtype.kind == "f"