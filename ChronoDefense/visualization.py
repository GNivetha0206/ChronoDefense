import matplotlib.pyplot as plt

def visualize_data(data):
    data['attack_type'].value_counts().plot(kind='bar')
    plt.xlabel('Attack Type')
    plt.ylabel('Frequency')
    plt.title('Distribution of Attack Types')
    plt.show()
