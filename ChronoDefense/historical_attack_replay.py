def replay_historical_attacks(data):
    print("Replaying historical attacks:")
    for index, row in data.iterrows():
        print(f"Attack Type: {row['attack_type']}, Affected System: {row['affected_system']}, Timestamp: {row['timestamp']}")
