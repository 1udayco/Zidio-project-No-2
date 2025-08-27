# 🔮 Customer Churn Prediction System

A comprehensive machine learning system for predicting customer churn using Flask web application and Jupyter notebooks.

## 📋 Project Overview

This project implements a customer churn prediction system that helps businesses identify customers who are likely to stop using their services. The system uses machine learning algorithms to analyze customer behavior patterns and provide actionable insights.

## 🚀 Features

- **Machine Learning Model**: Logistic Regression with high accuracy
- **Web Interface**: User-friendly Flask web application
- **Real-time Predictions**: Instant churn probability calculations
- **API Endpoint**: RESTful API for integration with other systems
- **Data Preprocessing**: Automated handling of missing values and feature scaling
- **Model Persistence**: Trained models saved for reuse

## 📊 Dataset

The system uses customer data with the following features:
- **Age**: Customer age
- **Gender**: Male/Female
- **Tenure**: Months as customer
- **Usage Frequency**: Service usage frequency
- **Support Calls**: Number of support calls
- **Payment Delay**: Days of payment delay
- **Subscription Type**: Basic/Standard/Premium
- **Contract Length**: Monthly/Quarterly/Annual
- **Total Spend**: Total amount spent
- **Last Interaction**: Days since last interaction

## 🛠️ Installation & Setup

### 1. Clone/Download the Project
```bash
# Navigate to project directory
cd "e:\3rd Project"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Windows PowerShell
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python test_models.py
```

## 🎯 Usage

### Web Application
1. **Start the Flask App**:
   ```bash
   python app.py
   ```

2. **Access the Application**:
   - Open your browser and go to: `http://localhost:5000`
   - Click "Start Prediction" to access the prediction form
   - Fill in customer details and get instant churn predictions

### API Usage
Send POST requests to `http://localhost:5000/api/predict` with JSON data:

```json
{
    "age": 35,
    "gender": "Male",
    "tenure": 24,
    "usage": 15,
    "support": 3,
    "delay": 5,
    "sub_type": "Premium",
    "contract": "Annual",
    "spend": 750.0,
    "last_interaction": 7
}
```

Response:
```json
{
    "prediction": 0,
    "churn_probability": 0.0214,
    "risk_level": "Low Risk"
}
```

### Jupyter Notebook
1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open Notebooks**:
   - `Notebook_Fixed.ipynb`: Complete analysis and model training
   - `Notebook.ipynb`: Original notebook (contains some issues)

## 📁 Project Structure

```
e:\3rd Project/
├── app.py                                    # Flask web application
├── test_models.py                           # Model testing script
├── requirements.txt                         # Python dependencies
├── README.md                               # Project documentation
├── Notebook.ipynb                          # Original notebook (has bugs)
├── Notebook_Fixed.ipynb                    # Fixed notebook
├── customer_churn_dataset-training-master.csv
├── customer_churn_dataset-testing-master.csv
├── model/                                  # Trained models directory
│   ├── churn_model.pkl                     # Trained logistic regression model
│   ├── scaler.pkl                          # Feature scaler
│   └── imputer.pkl                         # Missing value imputer
├── templates/                              # HTML templates
│   ├── index.html                          # Home page
│   └── predict.html                        # Prediction form
└── venv/                                   # Virtual environment
```

## 🔧 Model Details

### Algorithm
- **Model**: Logistic Regression
- **Preprocessing**: StandardScaler + SimpleImputer
- **Features**: 10 customer attributes
- **Performance**: High accuracy on test data

### Data Processing Pipeline
1. **Data Loading**: Read CSV files
2. **Categorical Encoding**: Label encoding for categorical variables
3. **Missing Value Handling**: Mean imputation
4. **Feature Scaling**: Standard scaling
5. **Model Training**: Logistic regression with regularization
6. **Model Evaluation**: Accuracy, precision, recall metrics

## 🐛 Bug Fixes Applied

### Original Notebook Issues Fixed:
1. **Double Imputation**: Removed redundant imputation steps
2. **Data Leakage**: Fixed preprocessing order
3. **Missing Evaluation**: Added comprehensive model evaluation
4. **Inconsistent Preprocessing**: Standardized the preprocessing pipeline
5. **Missing Feature Names**: Fixed feature name handling
6. **No Model Validation**: Added proper train/test evaluation

### Flask App Improvements:
1. **Error Handling**: Added comprehensive error handling
2. **Input Validation**: Added form validation
3. **User Experience**: Improved UI with better styling
4. **API Endpoint**: Added RESTful API for integration
5. **Model Loading**: Added proper model loading with error handling

## 📈 Performance Metrics

The model achieves:
- **High Accuracy**: >90% on test data
- **Good Precision**: Minimizes false positives
- **Good Recall**: Identifies most churning customers
- **Fast Predictions**: Real-time inference

## 🔍 Testing

### Automated Testing
```bash
python test_models.py
```

### Manual Testing
1. Use the web interface with sample data
2. Test API endpoints with curl or Postman
3. Run the Jupyter notebook cells

### Sample Test Cases
- **High Risk Customer**: Young, short tenure, many support calls
- **Low Risk Customer**: Older, long tenure, premium subscription

## 🚨 Troubleshooting

### Common Issues:

1. **Models Not Found**:
   - Run `python test_models.py` to train models
   - Check if `model/` directory exists

2. **Flask App Won't Start**:
   - Check if port 5000 is available
   - Verify all dependencies are installed

3. **Prediction Errors**:
   - Ensure all form fields are filled
   - Check data types match expected format

4. **Import Errors**:
   - Activate virtual environment
   - Install requirements: `pip install -r requirements.txt`

## 🔮 Future Enhancements

- [ ] Add more sophisticated ML models (Random Forest, XGBoost)
- [ ] Implement feature importance visualization
- [ ] Add batch prediction capability
- [ ] Create dashboard for churn analytics
- [ ] Add model retraining functionality
- [ ] Implement A/B testing for model comparison

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test script output
3. Examine Flask app logs
4. Verify data format matches expected schema

## 🎉 Success Indicators

✅ Virtual environment created and activated  
✅ All dependencies installed successfully  
✅ Models trained and saved  
✅ Flask app running on http://localhost:5000  
✅ Web interface accessible and functional  
✅ API endpoints responding correctly  
✅ Predictions working with sample data  
✅ All bugs from original notebook fixed  

---

**Happy Predicting! 🚀**