#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:17:28 2026

@author: neerajchoudhary
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Create sample customer data
# -----------------------------

data = {
    "monthly_spend": [50, 80, 30, 120, 60, 90, 25, 110, 45, 100],
    "support_calls": [1, 3, 0, 5, 2, 4, 0, 6, 1, 5],
    "months_active": [12, 6, 24, 3, 10, 5, 30, 2, 15, 4],
    "churned": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print("Customer Dataset:")
print(df)


# -----------------------------
# 2. Separate features and target
# -----------------------------

X = df[["monthly_spend", "support_calls", "months_active"]]
y = df["churned"]

print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)


# -----------------------------
# 3. Split data into train and test
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# -----------------------------
# 4. Train Logistic Regression model
# -----------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nModel training completed.")


# -----------------------------
# 5. Make predictions
# -----------------------------

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)

print("\nActual values:")
print(list(y_test))


# -----------------------------
# 6. Evaluate the model
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# 7. Predict for a new customer
# -----------------------------

new_customer = pd.DataFrame({
    "monthly_spend": [95],
    "support_calls": [4],
    "months_active": [5]
})

new_prediction = model.predict(new_customer)

print("\nNew Customer Prediction:")

if new_prediction[0] == 1:
    print("This customer is likely to churn.")
else:
    print("This customer is likely to stay.")