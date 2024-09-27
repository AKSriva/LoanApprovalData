# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:07:12 2024

@author: sandh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('loan_approval_dataset.csv')

# Cleaning categorical variables by stripping whitespace
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()

#shape of the data (row and columns)
df.shape

#get the description
df.describe()

#get data types
df.dtypes

#check the missing data
df.isnull().sum()
# Calculate percentage of missing values in each column
df.isnull().mean() * 100

# Heatmap to visualize missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()

# Target variable encoding (0 for "Rejected", 1 for "Approved")
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x == 'Approved' else 0)

# defining features and target
X = df.drop(columns=['loan_id', 'loan_status'])
y = df['loan_status']

# listing numerical and categorical features
numerical_features = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                      'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
categorical_features = ['education', 'self_employed']

# Data preprocessing pipeline for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer to apply preprocessing to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Stratified splitting to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess the training data before applying SMOTE
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)


# Model pipeline using XGBoost (without preprocessing step since we already preprocessed the data)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
model.fit(X_train_res, y_train_res)

# Preprocess the test data
X_test_preprocessed = preprocessor.transform(X_test)

# Ltets predict
y_pred = model.predict(X_test_preprocessed)


# Model evaluation
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
classification_rep = classification_report(y_test, y_pred)

# results matrix
results = {
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Confusion Matrix": conf_matrix,
    "Classification Report": classification_rep
}

results