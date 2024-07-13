# main.py
from data_preprocessing import preprocess_data
from anomaly_detection import detect_anomalies
from model_training import train_model, evaluate_model
from real_time_alerts import monitor_system

def main():
    # Load and preprocess the data
    data = preprocess_data("cyber_data.csv")

    # Split the data into features and labels
    features = data.drop("attack_success", axis=1)
    labels = data["attack_success"]

    # Train the model
    model = train_model(features, labels)

    # Evaluate the model
    evaluate_model(model, features, labels)

    # Detect anomalies
    anomalies = detect_anomalies(data)

    # Monitor system and send real-time alerts for detected anomalies
    monitor_system(anomalies)

if __name__ == "__main__":
    main()
