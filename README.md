Titanic Survival Prediction — Logistic Regression
Predicting whether a passenger survived the Titanic disaster using logistic regression.
Overview

Goal: Classify passengers as survived (1) or not survived (0)
Dataset: Titanic Dataset — Kaggle
Algorithm: Logistic Regression (scikit-learn)
Type: Binary classification

Results
MetricScoreAccuracyXX%Precision0.XXRecall0.XXF1 Score0.XX

Replace XX with your actual scores from the classification report in your notebook.

Features Used

Passenger class (Pclass)
Age
Number of siblings/spouses aboard (SibSp)
Number of parents/children aboard (Parch)
Fare
Embarked port

Project Structure
logistic-regression-classifier/
├── logistic_regression.ipynb   ← main notebook with full analysis
├── data.csv                    ← Titanic dataset
├── requirements.txt            ← required packages
└── README.md                   ← you are here
Key Findings

Female passengers had a significantly higher survival rate than male passengers
First class passengers were more likely to survive than second or third class
Age and fare also played a role in survival likelihood

Libraries Used

scikit-learn — logistic regression model and evaluation
pandas — data loading and preprocessing
numpy — numerical operations
matplotlib / seaborn — data visualisation

What I Learned

How to preprocess real-world data (handling missing values, encoding categorical variables)
How logistic regression works for binary classification
How to evaluate a classifier using accuracy, precision, recall, and F1 score
