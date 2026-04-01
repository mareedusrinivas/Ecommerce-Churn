#!/usr/bin/env python3
"""
Script to create an enhanced churn prediction model.
Handles categorical string data (Gender, Subscription, Contract) automatically.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def load_and_prepare_data():
    file_path = r'c:\Users\Srinivas\Downloads\customer_churn_dataset-testing-master.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    print(f"Loading and processing {file_path}...")
    df = pd.read_csv(file_path)

    # Dictionary-based encoding for known categorical columns
    mappings = {
        'Gender': {'Male': 0, 'Female': 1, 'M': 0, 'F': 1},
        'Subscription Type': {'Basic': 0, 'Standard': 1, 'Premium': 2},
        'Contract Length': {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            # Force conversion to string first to catch any mixed types
            df[col] = df[col].astype(str).str.strip()
            print(f"Applying mapping to {col}...")
            df[col] = df[col].map(mapping).fillna(0).astype(int)
            print(f"Successfully encoded {col}. Head: {df[col].head(3).tolist()}")

    # Fill any other missing values with 0
    df = df.fillna(0)
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == object and col != 'CustomerID':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                pass

    return df

def train_models(df):
    if df is None: return
    
    print("Training 13-feature Enhanced Model...")

    # Standard feature list for app consistency
    features = [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
        'Payment Delay', 'Subscription Type', 'Contract Length', 
        'Total Spend', 'Last Interaction', 'AnnualIncome', 'NumOrders', 'LastLoginDaysAgo'
    ]
    
    # Check if all features exist in DF
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Error: Missing columns in CSV: {missing}")
        return

    X = df[features]
    y = df['Churn']

    print(f"Final feature set: {X.columns.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Training Success! Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/nate_decision_tree.sav')
    print("Saved 'models/nate_decision_tree.sav'")

if __name__ == "__main__":
    data = load_and_prepare_data()
    train_models(data)
