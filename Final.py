import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# ----------------------------------------
# Dataset Loading Functions
# ----------------------------------------

def load_breast_cancer_data():
    columns = [
        "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
        "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
        "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ]
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        header=None, names=columns
    )
    data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])
    return data.iloc[:, 2:].values, data['diagnosis'].values

def load_ecoli_data():
    columns = ["sequence_name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
        header=None, names=columns, sep="\s+"
    )
    data['class'] = LabelEncoder().fit_transform(data['class'])
    return data.iloc[:, 1:-1].values, data['class'].values

def load_letter_recognition_data():
    columns = ["letter", "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar",
               "x2bar", "y2bar", "xybar", "x2ybar", "xy2bar", "x_ege", "xegvy", "y_ege", "yegvx"]
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        header=None, names=columns
    )
    data = data.sample(2000, random_state=42)
    data['letter'] = LabelEncoder().fit_transform(data['letter'])
    return data.iloc[:, 1:].values, data['letter'].values

def load_mushroom_data():
    columns = [
        "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor", "gill_attachment", "gill_spacing",
        "gill_size", "gill_color", "stalk_shape", "stalk_root", "stalk_surface_above_ring",
        "stalk_surface_below_ring", "stalk_color_above_ring", "stalk_color_below_ring", "veil_type",
        "veil_color", "ring_number", "ring_type", "spore_print_color", "population", "habitat"
    ]
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
        header=None, names=columns
    )
    for col in data.columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    return data.iloc[:, 1:].values, data['class'].values

# ----------------------------------------
# Model Functions
# ----------------------------------------

def evaluate_ann(X, y):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=1000, 
        learning_rate_init=0.0005, 
        early_stopping=True, 
        random_state=42
    )
    return evaluate_with_kfold(X, y, model, "ANN")

def evaluate_logistic_regression(X, y):
    model = LogisticRegression(
        max_iter=1000, 
        solver="saga", 
        random_state=42
    )
    return evaluate_with_kfold(X, y, model, "Logistic Regression")

def evaluate_naive_bayes(X, y):
    model = GaussianNB()
    return evaluate_with_kfold(X, y, model, "Naive Bayes")

def evaluate_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42
    )
    return evaluate_with_kfold(X, y, model, "Random Forest")

def evaluate_adaboost(X, y):
    model = AdaBoostClassifier(
        n_estimators=50, 
        random_state=42
    )
    return evaluate_with_kfold(X, y, model, "AdaBoost")

def evaluate_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=3)
    return evaluate_with_kfold(X, y, model, "KNN")

# ----------------------------------------
# Helper Functions
# ----------------------------------------

def evaluate_with_kfold(X, y, model, model_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if isinstance(model, (MLPClassifier, LogisticRegression)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    avg_accuracy = np.mean(accuracies)
    print(f" - {model_name} Accuracy: {avg_accuracy:.2f}")
    return avg_accuracy

# ----------------------------------------
# Main Script
# ----------------------------------------

if __name__ == "__main__":
    datasets = {
        "Breast Cancer": load_breast_cancer_data(),
        "E. coli": load_ecoli_data(),
        "Letter Recognition": load_letter_recognition_data(),
        "Mushroom": load_mushroom_data(),
    }

    for dataset_name, (X, y) in datasets.items():
        print(f"\nEvaluating {dataset_name} Dataset:")
        evaluate_ann(X, y)
        evaluate_logistic_regression(X, y)
        evaluate_naive_bayes(X, y)
        evaluate_random_forest(X, y)
        evaluate_adaboost(X, y)
        evaluate_knn(X, y)
