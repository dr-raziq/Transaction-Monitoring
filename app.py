import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_transactions, get_feature_matrix
from src.rule_engine import apply_rules, get_rule_summary
from src.anomaly_model import train_anomaly_model, score_anomalies
from src.scoring import combine_scores, get_alert_summary

st.set_page_config(page_title="Transaction Monitoring Engine", layout="wide")
st.title("Transaction Monitoring Engine")
st.caption("Rule-based detection combined with unsupervised anomaly detection on synthetic AML data.")

@st.cache_data
def load_and_process():
    df = load_transactions(nrows=100_000)  # limit for demo responsiveness
    df = apply_rules(df)
    features = get_feature_matrix(df)
    model, scaler = train_anomaly_model(features)
    scores = score_anomalies(model, scaler, features)
    df = pd.concat([df, scores], axis=1)
    df = combine_scores(df)
    return df

with st.spinner("Loading and scoring transactions..."):
    df = load_and_process()

summary = get_alert_summary(df)
rule_summary = get_rule_summary(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions analysed", f"{summary['total_transactions']:,}")
col2.metric("Rule alerts", f"{summary['rule_alerts']:,}")
col3.metric("ML anomalies", f"{summary['ml_alerts']:,}")
col4.metric("Total flagged", f"{summary['flagged_total']:,}")

st.subheader("Rule alert breakdown")
rule_df = pd.DataFrame([
    {"Rule": "Large transfer", "Alerts": rule_summary["large_transfer"]},
    {"Rule": "High frequency", "Alerts": rule_summary["high_frequency"]},
    {"Rule": "Structuring", "Alerts": rule_summary["structuring"]},
    {"Rule": "Rapid drain", "Alerts": rule_summary["rapid_drain"]},
])
fig_rules = px.bar(rule_df, x="Rule", y="Alerts", text="Alerts", color="Rule")
st.plotly_chart(fig_rules, use_container_width=True)

st.subheader("Risk distribution")
risk_counts = df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["Risk level", "Count"]
fig_risk = px.pie(risk_counts, names="Risk level", values="Count", hole=0.4)
st.plotly_chart(fig_risk, use_container_width=True)

st.subheader("Flagged transactions")
flagged = df[df["flagged"]].sort_values("anomaly_score")
show_cols = [
    "step", "type", "amount", "nameOrig", "nameDest",
    "rule_alert", "ml_alert", "risk_level", "anomaly_score"
]
st.dataframe(flagged[show_cols].head(100), use_container_width=True)

st.subheader("Anomaly score distribution")
fig_hist = px.histogram(df, x="anomaly_score", nbins=60, color="flagged")
st.plotly_chart(fig_hist, use_container_width=True)