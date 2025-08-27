from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load model, scaler, imputer
try:
    model = pickle.load(open("model/churn_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    imputer = pickle.load(open("model/imputer.pkl", "rb"))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = scaler = imputer = None

# Create label encoders based on the data structure
def create_label_encoders():
    """Create label encoders based on the training data"""
    encoders = {}
    
    # Gender encoder
    gender_encoder = LabelEncoder()
    gender_encoder.fit(['Female', 'Male'])
    encoders['Gender'] = gender_encoder
    
    # Subscription Type encoder
    sub_encoder = LabelEncoder()
    sub_encoder.fit(['Basic', 'Premium', 'Standard'])  # Alphabetical order for consistency
    encoders['Subscription Type'] = sub_encoder
    
    # Contract Length encoder
    contract_encoder = LabelEncoder()
    contract_encoder.fit(['Annual', 'Monthly', 'Quarterly'])  # Alphabetical order for consistency
    encoders['Contract Length'] = contract_encoder
    
    return encoders

label_encoders = create_label_encoders()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if not all([model, scaler, imputer]):
                return render_template('predict.html', 
                                     prediction="Error: Models not loaded properly. Please check model files.")
            
            # Get form inputs
            age = float(request.form['age'])
            gender = request.form['gender']  # 'Female' or 'Male'
            tenure = float(request.form['tenure'])
            usage = float(request.form['usage'])
            support = float(request.form['support'])
            delay = float(request.form['delay'])
            sub_type = request.form['sub_type']  # 'Basic', 'Standard', 'Premium'
            contract = request.form['contract']  # 'Monthly', 'Quarterly', 'Annual'
            spend = float(request.form['spend'])
            last_interaction = float(request.form['last_interaction'])

            # Encode categorical variables
            gender_encoded = label_encoders['Gender'].transform([gender])[0]
            sub_type_encoded = label_encoders['Subscription Type'].transform([sub_type])[0]
            contract_encoded = label_encoders['Contract Length'].transform([contract])[0]

            # Create feature array in the correct order
            # Order: Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, 
            #        Subscription Type, Contract Length, Total Spend, Last Interaction
            features = np.array([[age, gender_encoded, tenure, usage, support, delay,
                                  sub_type_encoded, contract_encoded, spend, last_interaction]])

            # Preprocess
            features = imputer.transform(features)
            features = scaler.transform(features)

            # Predict
            pred = model.predict(features)[0]
            pred_proba = model.predict_proba(features)[0]
            
            # Get probability of churn
            churn_probability = pred_proba[1] * 100  # Probability of class 1 (churn)
            
            if pred == 1:
                result = f"⚠️ Customer is likely to CHURN (Probability: {churn_probability:.1f}%)"
                risk_level = "High Risk"
            else:
                result = f"✅ Customer is NOT likely to churn (Churn Probability: {churn_probability:.1f}%)"
                risk_level = "Low Risk"

            return render_template('predict.html', 
                                 prediction=result, 
                                 risk_level=risk_level,
                                 probability=f"{churn_probability:.1f}%")

        except Exception as e:
            return render_template('predict.html', 
                                 prediction=f"Error: {str(e)}")

    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not all([model, scaler, imputer]):
            return jsonify({"error": "Models not loaded properly"}), 500
            
        data = request.get_json()
        
        # Extract features
        age = float(data['age'])
        gender = data['gender']
        tenure = float(data['tenure'])
        usage = float(data['usage'])
        support = float(data['support'])
        delay = float(data['delay'])
        sub_type = data['sub_type']
        contract = data['contract']
        spend = float(data['spend'])
        last_interaction = float(data['last_interaction'])

        # Encode categorical variables
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        sub_type_encoded = label_encoders['Subscription Type'].transform([sub_type])[0]
        contract_encoded = label_encoders['Contract Length'].transform([contract])[0]

        # Create feature array
        features = np.array([[age, gender_encoded, tenure, usage, support, delay,
                              sub_type_encoded, contract_encoded, spend, last_interaction]])

        # Preprocess
        features = imputer.transform(features)
        features = scaler.transform(features)

        # Predict
        pred = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        
        return jsonify({
            "prediction": int(pred),
            "churn_probability": float(pred_proba[1]),
            "risk_level": "High Risk" if pred == 1 else "Low Risk"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)