def feature_engineering(data):
    features = data[['attack_type', 'affected_system', 'attack_vector', 'timestamp']]
    labels = data['label']  # Assuming there's a 'label' column for attack types or severity
    return features, labels
