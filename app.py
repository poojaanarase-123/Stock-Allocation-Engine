from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
import joblib


app = Flask(__name__)
CORS(app)

# Load the saved Keras model
model = load_model('employee_risk_model.h5')

# Load the scaler and label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
stock_data = pd.read_csv('clusteredFinal22.csv')

def predict_cluster(age, salary, risk, model, scaler, label_encoder):
    risk_encoded = label_encoder.transform([risk])[0]
    input_data = pd.DataFrame([[age, salary, risk_encoded]], columns=['Age', 'Salary', 'Risk_encoded'])
    input_data_scaled = scaler.transform(input_data)
    cluster = model.predict(input_data_scaled)
    return int(cluster.argmax())

 

#Percentage Allocation
def filter_stocks(allocated_stocks, risk_tolerance):
    beta_min = allocated_stocks['Beta'].min()
    beta_max = allocated_stocks['Beta'].max()

    if risk_tolerance == 'low':
        threshold = beta_min + (beta_max - beta_min) / 3
        return allocated_stocks[allocated_stocks['Beta'] <= threshold]
    elif risk_tolerance == 'medium':
        lower_threshold = beta_min + (beta_max - beta_min) / 3
        upper_threshold = beta_min + 2 * (beta_max - beta_min) / 3
        return allocated_stocks[(allocated_stocks['Beta'] > lower_threshold) & (allocated_stocks['Beta'] <= upper_threshold)]
    else:  # high risk
        threshold = beta_min + 2 * (beta_max - beta_min) / 3
        return allocated_stocks[allocated_stocks['Beta'] > threshold]

def rank_stocks(allocated_stocks, target_savings):
    if target_savings > 1000000:
        allocated_stocks['Rank'] = allocated_stocks['Annual Return'].rank(ascending=False)
    else:
        allocated_stocks['Rank'] = allocated_stocks['Beta'].rank(ascending=True)
    return allocated_stocks.sort_values(by='Rank')

def allocate_percentage(allocated_stocks, num_stocks):
    selected_stocks = allocated_stocks.head(num_stocks).copy()
    total_rank = selected_stocks['Rank'].sum()
    
    selected_stocks['Allocation'] = selected_stocks['Rank'].apply(lambda rank: (total_rank - rank + 1) / total_rank)
    selected_stocks['Allocation'] = selected_stocks['Allocation'] / selected_stocks['Allocation'].sum() * 100
    return selected_stocks

def determine_number_of_stocks(salary, target_savings):
    if salary < 40000:
        return 2
    elif 40000 <= salary < 80000:
        return 3 + (1 if target_savings > 500000 else 0)
    else:
        return 5 + (1 if target_savings > 500000 else 0)


def generate_stocks(salary, age, risk_tolerance, target_savings, allocated_stocks):
        risk_tolerance = risk_tolerance.lower()
        filtered_data = filter_stocks(allocated_stocks, risk_tolerance)
        ranked_stocks = rank_stocks(filtered_data, target_savings)
    
        num_stocks = determine_number_of_stocks(salary, target_savings)
    
        finalStocks = allocate_percentage(ranked_stocks, num_stocks)
        return finalStocks

@app.route('/allocate', methods=['POST'])
def allocation():
    data = request.get_json()
    age = data.get('age')
    salary = data.get('salary')
    target_savings = data.get('target_savings')
    risk = data.get('risk_tolerance')
    predicted_cluster = predict_cluster(age, salary, risk, model, scaler, label_encoder)
    allocated_stocks = stock_data[stock_data['Cluster'] == predicted_cluster]
    allocated_per = generate_stocks(salary, age, risk, target_savings, allocated_stocks)
 
    
    
    response = {
        'predicted_cluster': predicted_cluster,
        'allocated_per': allocated_per.to_dict(orient='records'),
        'message': 'Success'
    }

    
    return jsonify(response)
 

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
