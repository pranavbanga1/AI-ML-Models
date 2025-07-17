import numpy as np
import pandas as pd
from collections import Counter

# Class to represent K-Nearest Neighbors
class KNN:
    def __init__(self, k=3, verbose=False):
        self.k = k  # Number of neighbors
        self.X_train = None
        self.y_train = None
        self.verbose = verbose  # Enable verbose mode for debugging

    # Fit function to store the training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Function to calculate Euclidean distance
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # Predict function for multiple samples
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    # Predict the class for a single sample
    def _predict_single(self, x):
        # Calculate distances from the current sample to all training samples
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Debug output to show distances and labels of k nearest neighbors
        if self.verbose:
            print("\nPredicting for sample:", x)
            print("Distances to neighbors:", distances)
            print("Indices of nearest neighbors:", k_indices)
            print("Labels of nearest neighbors:", k_nearest_labels)

        # Determine the most common label among the nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        if self.verbose:
            print("Predicted class:", most_common)
        return most_common

# Function to split the dataset into training and testing sets
def train_test_split(X, y, test_size=0.3, random_seed=42):
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    split_point = int((1 - test_size) * len(X))
    return X[indices[:split_point]], X[indices[split_point:]], y[indices[:split_point]], y[indices[split_point:]]

# Function to evaluate the model by calculating accuracy
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return np.mean(predictions == y_test)  # Accuracy = correct predictions / total predictions

#-------------------------------Dataset Loading Functions-------------------------------

# Letter Recognition Dataset
def load_letter_recognition_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
    columns = ["letter", "x_box", "y_box", "width", "height", "onpix", "x_bar", "y_bar", 
               "x2bar", "y2bar", "xybar", "x2ybar", "xy2bar", "xege", "xegvy", "yege", "yegvx"]
    data = pd.read_csv(url, header=None, names=columns)
    data['letter'] = pd.Categorical(data['letter']).codes
    X = data.iloc[:, 1:].values  # Features
    y = data['letter'].values  # Target
    return X, y

# Mushroom Dataset
def load_mushroom_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    columns = [
        "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor", "gill_attachment", "gill_spacing",
        "gill_size", "gill_color", "stalk_shape", "stalk_root", "stalk_surface_above_ring",
        "stalk_surface_below_ring", "stalk_color_above_ring", "stalk_color_below_ring", "veil_type",
        "veil_color", "ring_number", "ring_type", "spore_print_color", "population", "habitat"
    ]
    data = pd.read_csv(url, header=None, names=columns)
    for col in data.columns:
        data[col] = pd.Categorical(data[col]).codes  # Convert categorical to numeric codes
    X = data.iloc[:, 1:].values  # Features
    y = data['class'].values  # Target
    return X, y

# Ecoli Dataset
def load_ecoli_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
    columns = ["sequence_name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]
    data = pd.read_csv(url, header=None, names=columns)
    data['class'] = pd.Categorical(data['class']).codes  # Convert target to numeric codes
    X = data.iloc[:, 1:-1].values  # Features
    y = data['class'].values  # Target
    return X, y

# Breast Cancer Dataset
def load_breast_cancer_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = [
        "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
        "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst",
        "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    data = pd.read_csv(url, header=None, names=columns)
    data['diagnosis'] = pd.Categorical(data['diagnosis']).codes
    X = data.iloc[:, 2:].values  # Features (exclude id and diagnosis)
    y = data['diagnosis'].values  # Target
    return X, y

#-------------------------------Main Function-------------------------------

# Main function to test the KNN on multiple datasets with verbose mode
def main():
    print("\nLetter Recognition Dataset:")
    X, y = load_letter_recognition_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = KNN(k=5, verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nMushroom Dataset:")
    X, y = load_mushroom_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = KNN(k=5, verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nEcoli Dataset:")
    X, y = load_ecoli_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = KNN(k=5, verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nBreast Cancer Dataset:")
    X, y = load_breast_cancer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = KNN(k=5, verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
