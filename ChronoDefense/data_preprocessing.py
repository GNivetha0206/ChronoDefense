# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    return data
