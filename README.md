Transaction Monitoring Engine


What it does

    This project implements a transaction monitoring engine that combines deterministic AML rules with an unsupervised machine learning anomaly model. It processes synthetic mobile money transactions from the PaySim dataset, applies four configurable rules, trains an Isolation Forest to detect statistical outliers, and presents results through an interactive dashboard.

    The engine outputs:
    •	Rule alerts (large transfer, high frequency, structuring, rapid drain)
    •	ML anomaly scores and binary flags
    •	A combined risk level (LOW, MEDIUM, HIGH) for each transaction


Why

    Financial institutions in the European Union are required under the Anti-Money Laundering Regulation (AMLR) and related directives to implement ongoing transaction monitoring that can detect material deviations from expected behaviour [13†L3-L9]. Rule-based systems alone struggle to adapt to evolving patterns, while pure ML models can be opaque. This project demonstrates a hybrid approach that is both explainable and adaptive, using only synthetic data to avoid privacy concerns.

    The project is designed as a student portfolio piece to show practical understanding of AML compliance technology, feature engineering for financial data, and deployment of a monitoring dashboard.


Stack

    Python 3.10+
    pandas, numpy for data processing
    scikit-learn for Isolation Forest
    Streamlit for the dashboard
    Plotly for interactive charts
    pytest for testing


How to run

    1. Clone the repository
        git clone https://github.com/your-username/transaction-monitoring.git
        cd transaction-monitoring

    2. Install dependencies
        pip install -r requirements.txt

    3. Download the dataset
        Download PS_20174392719_1491204439457_log.csv from the PaySim Kaggle page and place it in the data/ directory. The dataset is licensed under CC BY-SA 4.0.

    4. Launch the dashboard
        streamlit run app.py
        Open the URL shown in the terminal (typically http://localhost:8501).

    5. Run tests
        pytest tests/ -v


Deployment
    The application is ready for deployment on Render or Railway free tiers.
    
    Render
        1.	Push the repository to GitHub.
        2.	In the Render dashboard, create a new Web Service.
        3.	Connect the repository.
        4.	Set the build command to pip install -r requirements.txt.
        5.	Set the start command to streamlit run app.py --server.port=$PORT --server.address=0.0.0.0.
        6.	Choose the Free instance type.
    Render free web services spin down after 15 minutes of inactivity and take about one minute to spin back up [14†L36-L40]. The filesystem is ephemeral, so the dataset must be included in the repository or downloaded at build time [14†L44-L49].

    Railway
        1.	Push the repository to GitHub.
        2.	In Railway, create a new project and select "Deploy from GitHub repo".
        3.	Railway auto-detects Python and uses the Procfile.
        4.	Set the environment variable STREAMLIT_SERVER_PORT=$PORT if needed.
    Railway's free plan provides $5 of credit for 30 days. The build memory is capped at 0.5 GB, which is sufficient for this application.


Project structure

    Transaction Monitoring
    |_ app.py                 
    |_ src
    |	|_ __init__.py
    |	|_ data_loader.py      
    |	|_ rule_engine.py      
    |	|_ anomaly_model.py    
    |	|_ scoring.py          
    |
    |_ tests
    |	|_ test_rule_engine.py
    |	|_ test_anomaly_model.py
    |                 
    |_ data
    |	|_ PS_20174392719_1491204439457_log.csv               
    |
    |_ requirements.txt
    |_ Procfile
    |_ .streamlit
    |	|_ config.toml
    |
    |_ LICENSE
    |_ .gitignore


Design decisions

    Rule engine. Four rules are implemented: large transfer threshold, high frequency per account per hour, structuring around reporting thresholds, and rapid drain of received funds. These rules are configurable and represent common AML typologies [10†L35-L44].
    
    Anomaly model. Isolation Forest is chosen because it is unsupervised, scales well to large datasets, and requires no labelled fraud data for training. It isolates outliers by randomly partitioning the feature space; anomalies require fewer splits and therefore have shorter path lengths [11†L40-L46]. The contamination parameter is set to 0.01, reflecting the low base rate of suspicious activity in the PaySim dataset (approximately 0.13%) [9†L3-L5].
    
    Combined scoring. A transaction is flagged if either the rule engine or the ML model raises an alert. Risk levels differentiate between single-source and dual-source alerts.
    
    EU compliance considerations. The design supports auditability by preserving the rule and ML alerts separately, allowing an investigator to understand why a transaction was flagged. No personal data is processed beyond the synthetic PaySim dataset. In a production system, the rule thresholds and contamination parameter would be calibrated to the institution's business-wide risk assessment and reviewed periodically, as required by AMLR Article 26(5) and related guidelines [13†L3-L9].

