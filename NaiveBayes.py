import numpy as np
import pandas as pd

# Naive Bayes Classifier
class NaiveBayes:
    def __init__(self, verbose=True):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
        self.verbose = verbose  # Enable verbose mode for debugging

    # Fit function calculates mean, variance, and prior probabilities
    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9  # Avoid division by zero
            self.priors[c] = X_c.shape[0] / X.shape[0]

    # Predict function for multiple samples
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    # Predicts the class for a single sample with detailed probability output
    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(np.log(self._gaussian_density(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

            # Verbose output to show probability details
            if self.verbose:
                print(f"\nClass {c}:")
                print(f"Prior log-probability: {prior}")
                print(f"Class conditional log-probability: {class_conditional}")
                print(f"Posterior log-probability: {posterior}")

        predicted_class = self.classes[np.argmax(posteriors)]
        if self.verbose:
            print(f"Predicted class: {predicted_class}")
        return predicted_class

    # Gaussian probability density function with smoothing
    def _gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        # Add a small constant (1e-9) to avoid zero probabilities
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var) + 1e-9 
        return numerator / denominator


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
    data = data.sample(n=1000, random_state=42).reset_index(drop=True)  # Randomly sample 1000 rows
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
    data = data.sample(n=1000, random_state=42).reset_index(drop=True)  # Randomly sample 1000 rows
    for col in data.columns:
        data[col] = pd.Categorical(data[col]).codes
    X = data.iloc[:, 1:].values  # Features
    y = data['class'].values  # Target
    return X, y

# Ecoli Dataset
def load_ecoli_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
    columns = ["sequence_name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]
    data = pd.read_csv(url, header=None, names=columns)
    data = data.sample(n=min(1000, len(data)), random_state=42).reset_index(drop=True)
    data['class'] = pd.Categorical(data['class']).codes
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
    data = data.sample(n=1000, random_state=42).reset_index(drop=True)
    data['diagnosis'] = pd.Categorical(data['diagnosis']).codes
    X = data.iloc[:, 2:].values  # Features
    y = data['diagnosis'].values  # Target
    return X, y

#-------------------------------Main Function-------------------------------

def main():
    results = []  #Summary

    print("\nLetter Recognition Dataset:")
    X, y = load_letter_recognition_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = NaiveBayes(verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    results.append(("Letter Recognition", accuracy))

    print("\nMushroom Dataset:")
    X, y = load_mushroom_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = NaiveBayes(verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    results.append(("Mushroom", accuracy))

    print("Ecoli Dataset:")
    X, y = load_ecoli_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = NaiveBayes(verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    results.append(("Ecoli", accuracy))

    print("\nBreast Cancer Dataset:")
    X, y = load_breast_cancer_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = NaiveBayes(verbose=True)
    model.fit(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    results.append(("Breast Cancer", accuracy))

    # Print summary of results
    print("\nSummary of Results:")
    for dataset_name, accuracy in results:
        print(f"{dataset_name} Dataset: Accuracy = {accuracy:.2f}")

if __name__ == "__main__":
    main()
