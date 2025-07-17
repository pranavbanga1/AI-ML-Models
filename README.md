# AI & Machine Learning Models – From Scratch 🧠💻

This repository contains machine learning algorithms implemented using only core Python libraries. It includes models for regression, classification, and tree-based methods, applied to real-world datasets such as Breast Cancer, Mushroom, E. coli, and Letter Recognition. Outputs and logs from the models are included to verify performance.

---

## 📁 Repository Structure

- `Lab2 Regression.py`: Implements Logistic and Linear Regression.
- `NaiveBayes.py`: Gaussian Naive Bayes classifier.
- `BreastCancerDecisionTree.py`: Custom decision tree model for classification.
- `DatasetSortingBreastCancer.py`: Dataset preprocessing and loading.
- `Test.py`: Model testing and evaluation.

---

## ✅ Features Implemented

### 🔷 Logistic Regression
- Gradient Descent optimization
- Accuracy reporting
- Used on Breast Cancer dataset

### 🔷 Linear Regression
- Cost function and gradient descent
- Mean Squared Error reporting
- Overflow handled manually in computation

### 🔷 Decision Tree Classifier
- Custom stump selection at each depth
- Built for Mushroom dataset
- Outputs tree structure and decision path

### 🔷 K-Nearest Neighbors (KNN)
- Euclidean distance calculation
- Predicts based on closest neighbors
- Outputs predicted class and neighbors' info

### 🔷 Naive Bayes Classifier
- Gaussian Naive Bayes from scratch
- Calculates class-conditional probabilities
- Accuracy displayed for each dataset

---

## 📊 Dataset Performance Summary

| Dataset               | Accuracy   |
|------------------------|------------|
| Breast Cancer          | 0.91 - 0.92|
| Mushroom               | 1.00       |
| E. coli                | 1.00       |
| Letter Recognition     | 0.61 - 0.64|

---

## 🖼️ Model Output Snapshots

Here are screenshots of model logs and terminal outputs included as proof of performance:

- 🧪 `Logistic Regression`: Accuracy and cost history plot (e.g., ~22.81% accuracy due to overflow)
- 📉 `Linear Regression`: Cost increases + NaN errors from overflow
- 🌲 `Decision Tree`: Tree depth and selected features per level
- 🔢 `KNN`: Neighbors and predicted class (code=1 success)
- 🧮 `Naive Bayes`: Class accuracies and warnings handled

Output Screenshots are stored in `/output_images/` or attached directly in the repo.

<img width="1284" height="518" alt="image" src="https://github.com/user-attachments/assets/ea772ad5-5b1c-4c39-8e61-303f85ea8509" />


<img width="1312" height="417" alt="image" src="https://github.com/user-attachments/assets/7f91d789-08e4-466d-969e-3a5a61572a17" />


<img width="935" height="349" alt="image" src="https://github.com/user-attachments/assets/8a2ee67f-d43a-44c0-a3fe-5b4d999588c0" />


<img width="1329" height="760" alt="image" src="https://github.com/user-attachments/assets/39f2e1cb-127a-4b2c-8124-72301e07ca49" />

---

## 🧠 Future Additions
- SVM from scratch
- Cross-validation utilities
- PCA / dimensionality reduction
- GUI for model testing

---

## 👤 Author

**Pranav Banga**  
📧 pranavbanga6@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/pranav-banga-a3756b200)  
🌐 [Portfolio](https://pranavbanga.netlify.app)

---

Feel free to fork and contribute! For any questions or feedback, reach out via email or LinkedIn.
