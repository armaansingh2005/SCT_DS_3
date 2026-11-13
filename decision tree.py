# ============================================================
# DECISION TREE CLASSIFIER - BANK MARKETING PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import zipfile
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

# Download ZIP file
zip_file = "bank.zip"

print("Downloading dataset...")

urllib.request.urlretrieve(url, zip_file)

# Extract ZIP
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall("bank_data")

print("Dataset downloaded successfully!")

# Load bank-full.csv
file_path = "bank_data/bank-full.csv"

df = pd.read_csv(file_path, sep=";")

# ------------------------------------------------------------
# 2. DISPLAY DATASET INFORMATION
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 3. TARGET VARIABLE
# ------------------------------------------------------------

# Target variable: y
# yes = customer subscribed/purchased
# no  = customer did not subscribe/purchase

print("\nTarget Distribution:")
print(df["y"].value_counts())

# ------------------------------------------------------------
# 4. ENCODE CATEGORICAL VARIABLES
# ------------------------------------------------------------

# Separate features and target
X = df.drop("y", axis=1)
y = df["y"]

# Convert categorical variables using One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Convert target:
# no  -> 0
# yes -> 1
y = y.map({"no": 0, "yes": 1})

print("\nEncoded Feature Shape:")
print(X.shape)

# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ------------------------------------------------------------
# 6. CREATE DECISION TREE CLASSIFIER
# ------------------------------------------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")

# ------------------------------------------------------------
# 7. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 8. MODEL ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

# ------------------------------------------------------------
# 9. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Purchase", "Purchase"]
))

# ------------------------------------------------------------
# 10. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Purchase", "Purchase"],
    yticklabels=["No Purchase", "Purchase"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# ------------------------------------------------------------
# 11. FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. VISUALIZE DECISION TREE
# ------------------------------------------------------------

plt.figure(figsize=(25, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Purchase", "Purchase"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier - Bank Marketing")
plt.show()

# ------------------------------------------------------------
# 13. PREDICT A NEW CUSTOMER
# ------------------------------------------------------------

# Example customer
new_customer = pd.DataFrame({
    "age": [35],
    "job": ["management"],
    "marital": ["married"],
    "education": ["tertiary"],
    "default": ["no"],
    "balance": [2500],
    "housing": ["yes"],
    "loan": ["no"],
    "contact": ["cellular"],
    "day": [15],
    "month": ["may"],
    "duration": [300],
    "campaign": [1],
    "pdays": [-1],
    "previous": [0],
    "poutcome": ["unknown"]
})

# One-hot encode the new customer
new_customer = pd.get_dummies(new_customer, drop_first=True)

# Make sure new customer has exactly the same columns as training data
new_customer = new_customer.reindex(
    columns=X.columns,
    fill_value=0
)

# Prediction
prediction = model.predict(new_customer)

# Probability
probability = model.predict_proba(new_customer)

print("\n======================================")
print("NEW CUSTOMER PREDICTION")
print("======================================")

if prediction[0] == 1:
    print("Prediction: CUSTOMER WILL PURCHASE / SUBSCRIBE")
else:
    print("Prediction: CUSTOMER WILL NOT PURCHASE / SUBSCRIBE")

print(
    "Probability of Purchase:",
    round(probability[0][1] * 100, 2),
    "%"
)

print(
    "Probability of No Purchase:",
    round(probability[0][0] * 100, 2),
    "%"
)