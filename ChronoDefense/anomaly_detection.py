# anomaly_detection.py
def detect_anomalies(data):
    # Simulated anomaly detection logic
    anomalies = data[data["attack_success"] == 1].to_dict(orient="records")
    return anomalies
