import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Column names for the dataset as provided in the dataset description
columns = ['ID', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
           'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 
           'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 
           'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
           'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
           'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# Load the dataset (without headers in the file, we apply the column names)
data = pd.read_csv(url, header=None, names=columns)

# Step 2: Prepare the data (Remove ID column and make 'Diagnosis' the target)
data = data.drop(columns=['ID'])  # Dropping the 'ID' column
data['Diagnosis'] = data['Diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convert 'M' (Malignant) to 1 and 'B' (Benign) to 0

# Step 3: Split the dataset into features (X) and target (y)
X = data.drop(columns=['Diagnosis'])  # All columns except 'Diagnosis'
y = data['Diagnosis']  # The 'Diagnosis' column is our target

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Output the shapes of the train and test sets
print("Training set (X_train) shape:", X_train.shape)
print("Training labels (y_train) shape:", y_train.shape)
print("Test set (X_test) shape:", X_test.shape)
print("Test labels (y_test) shape:", y_test.shape)

# Output the first 10 rows of the table with the Diagnosis column
print("\nFirst 10 rows of the dataset (including Diagnosis column):")
print(data.head(10))

# Output the distribution of the target variable in the training set
print("\nDistribution of target variable in training set (y_train):")
print(y_train.value_counts())
