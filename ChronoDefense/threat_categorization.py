def categorize_threats(data):
    categories = {
        'phishing': 'High',
        'malware': 'Medium',
        'ransomware': 'Critical',
        'ddos': 'High',
        'spyware': 'Medium'
    }
    data['label'] = data['attack_type'].map(categories)
    return data
