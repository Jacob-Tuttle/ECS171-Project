import joblib
import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

import logistic
from outliers import removeOutliers
import svm

import os
print("Current Working Directory:", os.getcwd())

st.header('ECS-171 Project - Classify Credit Score', divider='blue')

svmRBF = joblib.load("svm_rbf_model.pkl")
delay_from_due_date = st.slider("Delay from Due Date", 0, 50, 1)
num_of_delayed_payment = st.slider("Number of Delayed Payments", 0, 20, 1)
outstanding_debt = st.slider("Outstanding Debt", 0, 5000, 1)
credit_utilization_ratio = st.slider("Credit Utilization Ratio", 20, 50, 1)
credit_history_age = st.slider("Credit History Age", 10, 35, 1)

input = {
    'Delay_from_due_date': delay_from_due_date,
    'Num_of_Delayed_Payment': num_of_delayed_payment,
    'Outstanding_Debt': outstanding_debt,
    'Credit_Utilization_Ratio': credit_utilization_ratio,
    'Credit_History_Age': credit_history_age,
}

df = pd.DataFrame([input])
prediction = svmRBF.predict(df)

st.write(f"Predicted Credit Score: {prediction[0]}")
