#!/usr/bin/env python3
"""
Test script to verify the churn prediction models work correctly
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("🔍 Testing Customer Churn Prediction Models")
    print("=" * 50)
    
    # Check if model files exist
    model_files = [
        "model/churn_model.pkl",
        "model/scaler.pkl", 
        "model/imputer.pkl"
    ]
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing model files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n🔧 Training models from scratch...")
        train_models()
    else:
        print("✅ All model files found!")
    
    # Test the models
    test_models()
    
    print("\n🎉 Model testing completed!")

def train_models():
    """Train models if they don't exist"""
    try:
        # Read data
        train_df = pd.read_csv("customer_churn_dataset-training-master.csv")
        test_df = pd.read_csv("customer_churn_dataset-testing-master.csv")
        
        print(f"📊 Training data shape: {train_df.shape}")
        print(f"📊 Test data shape: {test_df.shape}")
        
        # Preprocess data
        X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)
        
        # Initialize and fit transformers
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_train_processed = imputer.fit_transform(X_train)
        X_train_processed = scaler.fit_transform(X_train_processed)
        
        X_test_processed = imputer.transform(X_test)
        X_test_processed = scaler.transform(X_test_processed)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_processed, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train_processed))
        test_acc = accuracy_score(y_test, model.predict(X_test_processed))
        
        print(f"📈 Training Accuracy: {train_acc:.4f}")
        print(f"📈 Test Accuracy: {test_acc:.4f}")
        
        # Save models
        os.makedirs("model", exist_ok=True)
        pickle.dump(imputer, open("model/imputer.pkl", "wb"))
        pickle.dump(scaler, open("model/scaler.pkl", "wb"))
        pickle.dump(model, open("model/churn_model.pkl", "wb"))
        
        print("💾 Models saved successfully!")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        raise

def preprocess_data(train_df, test_df):
    """Preprocess the data"""
    # Make copies
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # Encode categorical features
    cat_cols = ['Gender', 'Subscription Type', 'Contract Length']
    
    for col in cat_cols:
        le = LabelEncoder()
        train_processed[col] = le.fit_transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
    
    # Drop CustomerID
    train_processed.drop('CustomerID', axis=1, inplace=True)
    test_processed.drop('CustomerID', axis=1, inplace=True)
    
    # Split features and labels
    X_train = train_processed.drop('Churn', axis=1)
    y_train = train_processed['Churn']
    X_test = test_processed.drop('Churn', axis=1)
    y_test = test_processed['Churn']
    
    return X_train, y_train, X_test, y_test

def test_models():
    """Test the trained models"""
    try:
        # Load models
        model = pickle.load(open("model/churn_model.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        imputer = pickle.load(open("model/imputer.pkl", "rb"))
        
        print("✅ Models loaded successfully!")
        
        # Test with sample data
        sample_customers = [
            {
                'name': 'High Risk Customer',
                'data': [25, 0, 5, 2, 8, 25, 0, 0, 150, 25]  # Young, short tenure, high support calls, etc.
            },
            {
                'name': 'Low Risk Customer', 
                'data': [45, 1, 36, 20, 1, 2, 2, 2, 800, 3]   # Older, long tenure, premium, annual contract
            }
        ]
        
        print("\n🧪 Testing with sample customers:")
        print("-" * 40)
        
        for customer in sample_customers:
            features = np.array([customer['data']])
            
            # Preprocess
            features = imputer.transform(features)
            features = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            print(f"\n👤 {customer['name']}:")
            print(f"   Prediction: {'🔴 CHURN' if prediction == 1 else '🟢 NO CHURN'}")
            print(f"   Churn Probability: {probability[1]:.2%}")
            print(f"   Retention Probability: {probability[0]:.2%}")
        
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        raise

if __name__ == "__main__":
    main()