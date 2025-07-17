import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #Plot

#Plot Cost /Mean Square 
def plot_cost_vs_iterations(cost_history, label):
    plt.plot(cost_history, label=label)
    plt.xlabel("Iterations")
    plt.ylabel("Cost / Mean Square Error")
    plt.title(f"{label} vs. Iterations")
    plt.legend()
    plt.show()

# Function to load data from a given URL
def load_data(url, target_column=None):

    # Read data from the CSV
    data = pd.read_csv(url, header=None)

    # If target_column is specified, split it from the rest of the data
    if target_column is not None:
        X = data.drop(columns=[target_column]).values  # Features
        y = data[target_column].values  # Target
        return X, y
    else:
        # Return the entire dataset if target is not specified
        return data.values

# Function to split dataset into training and test sets
def split_dataset(X, y, test_size=0.2):

    # Calculate the split index based on the test_size
    split_point = int((1 - test_size) * len(X))
    
    # Split data into train and test sets
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    return X_train, X_test, y_train, y_test

# Normalize feature matrix (for both linear and logistic regression)
def normalize(X):
  
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Linear Regression Functions

# Training function for linear regression using gradient descent
def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Initialize weights to zero
    
    # Iterate to update theta values
    for i in range(iterations):
        # Calculate predictions using current theta values
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta -= learning_rate * gradient

    return theta

# Predict target values using trained linear regression model
def predict_linear(X, theta):
   
    return X.dot(theta)

# Mean Squared Error calculation
def mean_squared_error(y_true, y_pred):

    return np.mean(np.square(y_true - y_pred))

# R-squared score 
def r_squared(y_true, y_pred):
  
    ss_total = np.sum(np.square(y_true - np.mean(y_true)))
    ss_residual = np.sum(np.square(y_true - y_pred))
    return 1 - (ss_residual / ss_total)


# Logistic Regression Functions


# Sigmoid function for logistic regression
def sigmoid(z):
 
    z = np.clip(z, -500, 500)  # Manual Scaling
    return 1 / (1 + np.exp(-z))

# Training function for logistic regression using gradient descent
def logistic_regression(X, y, learning_rate=0.001, iterations=1000):

    m, n = X.shape
    theta = np.zeros(n)  # Initialize weights to zero
    
    for i in range(iterations): #Update Theta
        predictions = sigmoid(X.dot(theta))
        # Compute error and gradient
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        # Update theta
        theta -= learning_rate * gradient
        
    return theta

# Compute the cost for logistic regression
def compute_logistic_cost(X, y, theta):

   # logistic regression cost function.
    predictions = sigmoid(X.dot(theta))
    m = len(y)
    return -(1/m) * np.sum(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15)) #Lecture Notes

# Predict class labels using logistic regression
def predict_logistic(X, theta, threshold=0.5):
    
    probabilities = sigmoid(X.dot(theta))
    return (probabilities >= threshold).astype(int)

# Accuracy calculation for classification
def accuracy(y_true, y_pred):
  
    return np.mean(y_true == y_pred) * 100

#Main

if __name__ == "__main__":
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    X, y = load_data(dataset_url, target_column=1)  # Load features and target

    # Convert target to binary: 'M' (malignant) -> 1, 'B' (benign) -> 0
    y = np.where(y == 'M', 1, 0)
    
    # Normalize the feature data
    X = normalize(X)
    
    # Add bias term (column of ones) to X
    X = np.c_[np.ones(X.shape[0]), X]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Logistic Regression
    theta_logistic = logistic_regression(X_train, y_train, learning_rate=0.01, iterations=1000)
    y_train_pred = predict_logistic(X_train, theta_logistic)
    
    # Output results for logistic regression
    final_cost_logistic = compute_logistic_cost(X_train, y_train, theta_logistic)
    print(f"Logistic Regression Final Cost: {final_cost_logistic}")
    print(f"Logistic Regression Training Accuracy: {accuracy(y_train, y_train_pred):.2f}%")
    print (f"First 5 outputs",y_train_pred[:5])

    # Linear Regression
    theta_linear = linear_regression(X_train, y_train, learning_rate=0.01, iterations=1000)
    y_test_pred_linear = predict_linear(X_test, theta_linear)
    
    # Output results for linear regression
    test_mse = mean_squared_error(y_test, y_test_pred_linear)
    test_r2 = r_squared(y_test, y_test_pred_linear)
    print(f"Linear Regression MSE: {test_mse}")
    print(f"Linear Regression R-squared: {test_r2}")
    print (f"First 5 outputs",y_test_pred_linear[:5])

  # Plot Mean Squared Error vs. Iterations
    #plot_cost_vs_iterations(cost_history, label="Linear MSE")