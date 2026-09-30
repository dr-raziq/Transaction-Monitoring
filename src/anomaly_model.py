import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def train_anomaly_model(
    feature_matrix: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[IsolationForest, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler

def score_anomalies(
    model: IsolationForest,
    scaler: StandardScaler,
    feature_matrix: pd.DataFrame,
) -> pd.DataFrame:
    X_scaled = scaler.transform(feature_matrix)
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    result = pd.DataFrame({
        "anomaly_score": scores,
        "anomaly_pred": predictions,
    })
    result["ml_alert"] = result["anomaly_pred"] == -1
    return result