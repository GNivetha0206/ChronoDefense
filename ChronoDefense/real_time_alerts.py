# real_time_alerts.py
import time

def send_alert(alert_message):
    """
    Simulate sending a real-time alert.
    This function could be expanded to send emails, SMS, etc.
    """
    print(f"ALERT: {alert_message}")

def monitor_system(anomalies):
    """
    Monitor the system for anomalies and send real-time alerts.
    """
    for anomaly in anomalies:
        # Customize the alert message based on anomaly details
        alert_message = f"Anomaly detected: {anomaly}"
        send_alert(alert_message)
        # Simulate a delay between alerts
        time.sleep(1)

# Example function usage
if __name__ == "__main__":
    # Sample anomalies data
    anomalies = [
        {"attack_type": "DDoS", "affected_system": "Web Server", "attack_vector": "Network", "timestamp": "2024-07-10 10:00:00"},
        {"attack_type": "Phishing", "affected_system": "Email Server", "attack_vector": "Social Engineering", "timestamp": "2024-07-10 10:05:00"}
    ]
    monitor_system(anomalies)
