# model_training.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, features, labels):
    # Evaluate the model on the entire dataset
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f"Overall Model Accuracy: {accuracy}")
    print("Overall Classification Report:")
    print(classification_report(labels, predictions))

    # Evaluate the model on a test set
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Set Accuracy: {test_accuracy}")
    print("Test Set Classification Report:")
    print(classification_report(y_test, test_predictions))