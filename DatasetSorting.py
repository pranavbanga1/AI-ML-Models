import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# The dataset does not have headers, so we need to specify the column names manually
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load the dataset
data = pd.read_csv(url, names=columns)

# Step 2: Place the target variable (target) as the last column
def prepare_data(data, target_column):
    target = data.pop(target_column)
    data[target_column] = target
    return data

data = prepare_data(data, 'target')

# Step 3: Split the dataset into training and testing sets
def split_data(data, test_size=0.2):
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=42, shuffle=True)
    return train_set, test_set

train_set, test_set = split_data(data)

# Display the result sizes
print(f"Train set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")

# Output the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Output the shape of the dataset (rows, columns)
print("\nDataset shape:")
print(data.shape)

# Output the distribution of the target variable
print("\nDistribution of the target variable:")
print(data['target'].value_counts())