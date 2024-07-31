import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

stock_data = pd.read_csv('clusteredFinal22.csv')
employee_data = pd.read_csv('employee_risk_data_with_outliers.csv')

label_encoder = LabelEncoder()
employee_data['Risk_encoded'] = label_encoder.fit_transform(employee_data['Risk'])


features = ['Age', 'Salary', 'Risk_encoded']

def map_risk_to_cluster(risk):
    if risk == 'low':
        return 2 if np.random.rand() < 0.5 else 3
    elif risk == 'medium':
        return 4
    elif risk == 'high':
        return 1 if np.random.rand() < 0.5 else 0
np.random.seed(0) 
employee_data['Cluster'] = employee_data['Risk'].apply(map_risk_to_cluster)
employee_data.head()


X = employee_data[features]
y = employee_data['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(32, input_dim=len(features), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, validation_split=0.2)

import joblib

model.save('employee_risk_model.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')


