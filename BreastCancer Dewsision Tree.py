import numpy as np
import pandas as pd

#Class to evaluate and analyse data into a decision tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Overfiting Max Depth
        self.tree = None  # The tree will be built and stored here
        self.feature_names = []  # To store the feature names

    # Fit function builds the tree using the training data X and labels y
    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names if feature_names is not None else []
        self.tree = self._build_tree(X, y)

    # This function recursively builds the decision tree
    def _build_tree(self, X, y, state=0):
        num_samples, num_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # Stop condition: If all samples belong to the same class or max depth is reached
        if len(unique_classes) == 1 or (self.max_depth and state >= self.max_depth):
            print(f"Reached leaf node: Class {unique_classes[np.argmax(class_counts)]}")
            return unique_classes[np.argmax(class_counts)]  # Return the majority class

        # Find the feature that gives the highest information gain
        gains = np.array([self._information_gain(y, X[:, feature]) for feature in range(num_features)])
        best_feature = np.argmax(gains)

        # Get feature name for printing
        feature_name = self.feature_names[best_feature] if self.feature_names else f"Feature {best_feature}"

        # Print the chosen stump (best feature)
        print(f"Chose feature '{feature_name}' as the stump at state {state}")

        # Create a tree structure with the best feature
        tree = {best_feature: {}}
        for value in np.unique(X[:, best_feature]):
            # Subset the data for the feature value and recursively build subtrees
            indices = X[:, best_feature] == value
            subtree = self._build_tree(X[indices], y[indices], state + 1)
            tree[best_feature][value] = subtree

        return tree

    # Function to calculate information gain (how well a feature splits the data)
    def _information_gain(self, y, feature_column):
        # Entropy before the split
        entropy_before = self._entropy(y)
        unique_values, counts = np.unique(feature_column, return_counts=True)

        # Calculate weighted entropy after the split
        weighted_entropy_after = sum(
            (counts[i] / len(feature_column)) * self._entropy(y[feature_column == unique_values[i]])
            for i in range(len(unique_values))
        )

        return entropy_before - weighted_entropy_after

    # Function to calculate entropy (a measure of uncertainty)
    def _entropy(self, y):
        probabilities = [np.sum(y == c) / len(y) for c in np.unique(y)]
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    # Function to make predictions on new data
    def predict(self, X):
        return np.array([self._predict_single(sample, self.tree) for sample in X])

    # This function navigates the tree for a single sample to predict the output
    def _predict_single(self, sample, tree):
        if not isinstance(tree, dict):  # If it's a leaf node, return the class
            return tree
        feature = next(iter(tree))  # The feature index to split on
        feature_value = sample[feature]  # Get the value of that feature for the sample
        if feature_value in tree[feature]:  # Follow the branch that matches the value
            return self._predict_single(sample, tree[feature][feature_value])
        return None  # Return None if the value wasn't seen during training

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

#------------------------------------------------Dataset----------------------------------------------------------------
#Letter Recognisation
def load_letter_recognition_data():
          
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
    
    # Defining column names
    #columns = ["letter"] + [f"feature_{i}" for i in range(1, 17)]  # 16 features plus the target 'letter' // If no name are required 
    columns = ["letter", "x_box", "y_box", "width", "height", "onpix", "x_bar", "y_bar", 
               "x2bar", "y2bar", "xybar", "x2ybar", "xy2bar", "xege", "xegvy", "yege", "yegvx"]
    # Loading the dataset using pandas
    data = pd.read_csv(url, header=None, names=columns)
    # Convert the target 'letter' to numeric codes (A=0, B=1, ..., Z=25)
    data['letter'] = pd.Categorical(data['letter']).codes
    # Split into features (X) and target (y)
    X = data.iloc[:, 1:].values  # Features (all columns except 'letter')
    y = data['letter'].values  # Target (letter A-Z)
    return X, y, columns[1:]

#  Mushroom dataset
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
        data[col] = pd.Categorical(data[col]).codes  # Convert categorical data to numeric codes
    X = data.iloc[:, 1:].values  # Features
    y = data['class'].values  # Target (edibility)
    return X, y, columns[1:]  # Return features, target, and feature names

#  Ecoli dataset
def load_ecoli_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
    columns = ["sequence_name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]
    data = pd.read_csv(url, header=None, names=columns)
    data['class'] = pd.Categorical(data['class']).codes  # Convert target to numeric codes
    X = data.iloc[:, 1:-1].values  # Features
    y = data['class'].values  # Target (protein localization site)
    return X, y, columns[1:-1]  # Return features, target, and feature names

# Breast Cancer dataset
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
    # Convert diagnosis to numeric (0=benign, 1=malignant)
    data['diagnosis'] = pd.Categorical(data['diagnosis']).codes
    # Extract features and target
    X = data.iloc[:, 2:].values  # Features (exclude id and diagnosis)
    y = data['diagnosis'].values  # Target (diagnosis: benign or malignant)
    return X, y, columns[2:]  # Return features, target, and feature names

#------------------------------------------Main------------------------------------------------------------------------

# Main function to test the decision tree on multiple datasets
def main():

        
    print("\nLetter Recognition Dataset:")
    X, y, feature_names = load_letter_recognition_data()  # Load the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # Split the dataset
    model = DecisionTree(max_depth=None)
    model.fit(X_train, y_train, feature_names)  # Train the decision tree
    accuracy = evaluate_model(model, X_test, y_test)  # Evaluate the model
    print(f"Accuracy: {accuracy:.2f}")  # Print the accuracy

    
    print("\nMushroom Dataset:")
    X, y, feature_names = load_mushroom_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = DecisionTree(max_depth=None)
    model.fit(X_train, y_train, feature_names)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

   
    print("\nEcoli Dataset:")
    X, y, feature_names = load_ecoli_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train, feature_names)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

   
    print("\nBreast Cancer Dataset:")
    X, y, feature_names = load_breast_cancer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train, feature_names)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
