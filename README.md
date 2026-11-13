# 🌳 Decision Tree Classifier – Bank Marketing Prediction

## 📌 Project Overview

This project implements a **Decision Tree Classification Machine Learning Model** using the Bank Marketing Dataset from the UCI Machine Learning Repository. The model predicts whether a customer will subscribe to a bank's term deposit based on demographic, financial, and marketing campaign-related information.

The project demonstrates data preprocessing, categorical encoding, model training, prediction, performance evaluation, and data visualization using Python and Scikit-learn.

## 🎯 Objectives

- Build a Decision Tree Classification model.
- Analyze customer information and marketing campaign data.
- Predict whether a customer will subscribe to a term deposit.
- Evaluate model performance using classification metrics.
- Identify important features influencing predictions.
- Visualize the decision tree and confusion matrix.

## 📊 Dataset

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository

The dataset contains customer information and details about direct marketing campaigns conducted by a Portuguese banking institution.

**Target Variable:**
- `yes` – Customer subscribed to a term deposit.
- `no` – Customer did not subscribe to a term deposit.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## ⚙️ Project Workflow

1. **Data Collection:** Download and extract the Bank Marketing Dataset.
2. **Data Exploration:** Analyze dataset structure, columns, and missing values.
3. **Data Preprocessing:** Apply One-Hot Encoding to categorical features.
4. **Target Encoding:** Convert target labels into numerical values.
5. **Train-Test Split:** Divide the dataset into 80% training and 20% testing data.
6. **Model Building:** Implement a Decision Tree Classifier using the Gini criterion.
7. **Model Training:** Train the classifier using the training dataset.
8. **Prediction:** Predict customer subscription outcomes.
9. **Model Evaluation:** Evaluate performance using accuracy, classification report, and confusion matrix.
10. **Visualization:** Display feature importance and decision tree structure.

## 🤖 Machine Learning Algorithm

**Decision Tree Classifier**

The model uses:
- Criterion: Gini
- Maximum Depth: 5
- Random State: 42

The Decision Tree algorithm makes predictions by splitting the dataset into smaller subsets based on feature values.

## 📈 Model Evaluation

The model evaluates its predictions using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

Additional visualizations include:
- Confusion Matrix Heatmap
- Top 15 Feature Importance Chart
- Decision Tree Visualization

## 🔍 New Customer Prediction

The project also demonstrates prediction for a new customer using sample demographic, financial, and campaign-related information.

The model provides:
- Customer subscription prediction.
- Probability of subscription.
- Probability of non-subscription.

## 🚀 Installation & Execution

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project directory**

```bash
cd YOUR_PROJECT_FOLDER
```

**3. Install required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**4. Run the Python file**

```bash
python "decision tree.py"
```

The dataset is downloaded automatically from the UCI Machine Learning Repository when the script runs.

## 💡 Key Learning Outcomes

- Understanding Decision Tree Classification.
- Working with real-world datasets.
- Performing categorical data encoding.
- Training and testing machine learning models.
- Evaluating classification performance.
- Understanding feature importance.
- Visualizing machine learning models.
- Making predictions for new customer data.

## 🔮 Future Improvements

- Compare Decision Tree with Random Forest, SVM, and Logistic Regression.
- Perform hyperparameter tuning.
- Improve model performance through feature engineering.
- Develop an interactive web-based prediction dashboard.
- Deploy the model using Flask or Streamlit.

## 👨‍💻 Author

**Armaan Singh**

B.Tech – Artificial Intelligence & Machine Learning  
Dronacharya College of Engineering, Gurugram

---

⭐ If you find this project useful, consider giving this repository a star!
