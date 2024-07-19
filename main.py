import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random
import socket
import geocoder  # Ensure geocoder package is installed: pip install geocoder

# Function to get location (latitude, longitude) from IP address
def get_location_from_ip(ip):
    g = geocoder.ip(ip)
    if g.ok:
        return g.latlng
    else:
        return None

# Simulate historical cyber attack data based on system's IP addresses
def generate_data(num_records=1000):
    np.random.seed(42)
    random.seed(42)
    
    # Get system's IP address
    system_ip = socket.gethostbyname(socket.gethostname())
    
    # Get system's location based on IP address
    system_location = get_location_from_ip(system_ip)
    if system_location:
        system_latitude, system_longitude = system_location
        system_region = determine_region(system_latitude, system_longitude)
    else:
        # Default region if location detection fails
        system_region = 'Unknown'
    
    attack_types = ['DDoS', 'Phishing', 'Malware', 'Ransomware']
    ports = [80, 443, 21, 22, 25, 8080]
    protocols = ['TCP', 'UDP', 'ICMP']
    
    data = []
    
    for _ in range(num_records):
        ip = random.choice([system_ip])  # Use system's IP address for all records
        attack_type = random.choice(attack_types)
        port = random.choice(ports)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))
        attack_duration = random.randint(1, 3600)  # in seconds
        region = system_region  # Use system's region for all records
        impact = random.randint(1, 10)
        success_level = random.randint(1, 10)
        protocol = random.choice(protocols)
        
        severity_score = impact * success_level  # Simplified severity score calculation
        
        data.append([severity_score, ip, attack_type, port, timestamp, attack_duration, region, impact, success_level, protocol])
    
    columns = ['SeverityScore', 'IPAddress', 'AttackType', 'Port', 'Timestamp', 'AttackDuration', 'Region', 'Impact', 'SuccessLevel', 'Protocol']
    df = pd.DataFrame(data, columns=columns)
    
    return df

# Function to determine region based on latitude and longitude
def determine_region(latitude, longitude):
    # Implement your region determination logic based on coordinates
    # Example logic: for demo, assuming a simple division based on latitude
    if latitude > 0:
        return 'US'  # Example region determination for Northern Hemisphere
    else:
        return 'EU'  # Example region determination for Southern Hemisphere

# Generate and save historical data
historical_data = generate_data()
historical_data.to_csv('historical_cyber_attacks.csv', index=False)

# Load historical data
data = pd.read_csv('historical_cyber_attacks.csv')

# Preprocess data
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = pd.get_dummies(data, columns=['AttackType', 'Region', 'Protocol'])

# Save original columns before converting to NumPy array
original_columns = data.drop(columns=['SeverityScore', 'IPAddress', 'Timestamp']).columns

# Split data into features and labels
X = data.drop(columns=['SeverityScore', 'IPAddress', 'Timestamp'])
y = data['SeverityScore']

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Predict future threats (simulated data for the next month)
def predict_future_threats(model, scaler, original_columns, num_days=30):
    future_data = generate_data(num_records=num_days)
    future_data['Timestamp'] = pd.to_datetime(future_data['Timestamp'])
    future_data = pd.get_dummies(future_data, columns=['AttackType', 'Region', 'Protocol'])
    
    # Ensure future data has same columns as training data
    for col in original_columns:
        if col not in future_data.columns:
            future_data[col] = 0
    future_data = future_data[original_columns]
    
    future_X = scaler.transform(future_data)
    future_data['PredictedSeverityScore'] = model.predict(future_X)
    
    return future_data

future_threats = predict_future_threats(model, scaler, original_columns)

# Save future threats predictions to CSV
future_threats.to_csv('predicted_future_threats.csv', index=False)

# Generate mitigation measures (simplified example)
def suggest_mitigation_measures(threats):
    mitigation_measures = []
    for _, row in threats.iterrows():
        if row['PredictedSeverityScore'] > 70:
            measure = "Immediate action required: Isolate the affected systems and begin incident response procedures."
        elif row['PredictedSeverityScore'] > 40:
            measure = "High priority: Monitor the systems closely and prepare for potential incident response."
        else:
            measure = "Low priority: Regular monitoring and standard security practices."
        mitigation_measures.append(measure)
    
    threats['MitigationMeasures'] = mitigation_measures
    return threats

future_threats_with_measures = suggest_mitigation_measures(future_threats)

# Save final report to CSV
future_threats_with_measures.to_csv('cyber_threats_report.csv', index=False)

print("Cyber threats report generated successfully.")
