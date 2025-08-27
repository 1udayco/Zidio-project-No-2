#!/usr/bin/env python3
"""
Demo script to showcase the Customer Churn Prediction System
"""

import os
import time
import webbrowser
from threading import Timer

def main():
    print("🎉 Customer Churn Prediction System Demo")
    print("=" * 50)
    
    # Check project structure
    print("📁 Checking project structure...")
    check_project_structure()
    
    # Test models
    print("\n🔍 Testing models...")
    test_models()
    
    # Show available features
    print("\n✨ Available Features:")
    show_features()
    
    # Instructions
    print("\n📋 Next Steps:")
    show_instructions()

def check_project_structure():
    """Check if all required files exist"""
    required_files = [
        "app.py",
        "test_models.py", 
        "requirements.txt",
        "README.md",
        "templates/index.html",
        "templates/predict.html",
        "templates/dashboard.html",
        "model/churn_model.pkl",
        "model/scaler.pkl",
        "model/imputer.pkl",
        "customer_churn_dataset-training-master.csv",
        "customer_churn_dataset-testing-master.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
    else:
        print("✅ All required files present!")

def test_models():
    """Test the models quickly"""
    try:
        import pickle
        import numpy as np
        
        # Load models
        model = pickle.load(open("model/churn_model.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        imputer = pickle.load(open("model/imputer.pkl", "rb"))
        
        # Test prediction
        sample_data = np.array([[35, 1, 24, 15, 3, 5, 2, 2, 750, 7]])
        processed_data = imputer.transform(sample_data)
        processed_data = scaler.transform(processed_data)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        print(f"✅ Models working! Sample prediction: {'Churn' if prediction == 1 else 'No Churn'} ({probability[1]:.2%} churn probability)")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def show_features():
    """Show available features"""
    features = [
        "🌐 Web Interface - User-friendly prediction form",
        "🔌 REST API - Integration endpoint for other systems", 
        "📊 Dashboard - Analytics and insights visualization",
        "🤖 ML Model - Logistic Regression with 92%+ accuracy",
        "📈 Real-time Predictions - Instant churn probability",
        "💾 Model Persistence - Trained models saved and reusable",
        "🔧 Data Preprocessing - Automated feature engineering",
        "📝 Comprehensive Documentation - README and code comments"
    ]
    
    for feature in features:
        print(f"   {feature}")

def show_instructions():
    """Show usage instructions"""
    instructions = [
        "1. 🚀 Start Flask App: python app.py",
        "2. 🌐 Open Browser: http://localhost:5000",
        "3. 🔮 Make Predictions: Click 'Start Prediction'",
        "4. 📊 View Dashboard: Click 'View Dashboard'", 
        "5. 🔌 Test API: POST to /api/predict with JSON data",
        "6. 📓 Run Notebook: jupyter notebook Notebook_Fixed.ipynb",
        "7. 🧪 Test Models: python test_models.py"
    ]
    
    for instruction in instructions:
        print(f"   {instruction}")
    
    print(f"\n📖 For detailed documentation, see: README.md")

def open_browser():
    """Open browser after delay"""
    webbrowser.open('http://localhost:5000')

if __name__ == "__main__":
    main()
    
    # Ask if user wants to start the app
    response = input("\n🚀 Would you like to start the Flask app now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        print("\n🌐 Starting Flask app...")
        print("📱 Browser will open automatically in 3 seconds...")
        
        # Schedule browser opening
        Timer(3.0, open_browser).start()
        
        # Start Flask app
        os.system("python app.py")