def suggest_mitigations(data):
    suggestions = {
        'phishing': 'Implement multi-factor authentication and user training.',
        'malware': 'Install and update antivirus software.',
        'ransomware': 'Regularly backup data and avoid clicking on suspicious links.',
        'ddos': 'Deploy DDoS protection services and monitor network traffic.',
        'spyware': 'Use anti-spyware tools and regularly scan for spyware.'
    }
    data['mitigation'] = data['attack_type'].map(suggestions)
    return data
