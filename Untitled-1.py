# ==============================
# Traffic Accident Severity Prediction
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("traffic_accidents.csv")

print("Dataset Preview")
print(df.head())


# ==============================
# Check Duplicates
# ==============================
print("\nDuplicate Rows:", df.duplicated().sum())


# ==============================
# Dataset Information
# ==============================
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())


# ==============================
# Handle Missing Values
# ==============================
df.dropna(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ==============================
# Drop Unnecessary Columns
# ==============================
df = df.drop([
    'crash_date',
    'injuries_total',
    'injuries_fatal',
    'injuries_incapacitating',
    'injuries_non_incapacitating',
    'injuries_reported_not_evident'
], axis=1)


# ==============================
# Define Features and Target
# ==============================
X = df.drop('most_severe_injury', axis=1)
y = df['most_severe_injury']


# ==============================
# Feature Encoding
# ==============================
X = pd.get_dummies(X, drop_first=True)


# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ==============================
# Feature Scaling
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# Logistic Regression Model
# ==============================
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',
    max_iter=2000,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)


# ==============================
# Predictions
# ==============================
y_pred = model.predict(X_test_scaled)


# ==============================
# Model Evaluation
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ==============================
# Confusion Matrix Visualization
# ==============================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.title("Confusion Matrix - Logistic Regression")
plt.show()