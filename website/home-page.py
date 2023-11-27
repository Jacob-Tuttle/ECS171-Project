import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

st.header('ECS-171 Project - Classify Credit Score', divider='blue')

uploaded_file = st.file_uploader("Upload CSV File")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))


    # To read file as string:
    string_data = stringio.read()


    # Can be used wherever a "file-like" object is accepted:https://github.com/Jacob-Tuttle/ECS171-Project/blob/Website/website/home-page.py
    data = pd.read_csv(uploaded_file)
    
    drop = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour',
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',  'Num_Credit_Inquiries',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Num_of_Delayed_Payment']

    # Columns to clean
    clean = ['Delay_from_due_date', 'Monthly_Inhand_Salary', 'Monthly_Balance', 'Changed_Credit_Limit','Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age']

        # Clean columns
    for item in clean:
        data[item] = pd.to_numeric(data[item], errors='coerce')
    
    # Map credit scores to number
    creditScoreMap = {'Poor': 1, 'Standard': 2, 'Good': 3}
    data['Credit_Score'] = data['Credit_Score'].replace(creditScoreMap)
    
    
    cleanedData = train.copy().drop(columns=drop)
    # Drop entries with NaN values
    cleanedData.dropna(inplace=True)
    # Separate features (X) and target variable (y)
    X = cleanedData.drop(columns=['Credit_Score'])
    y = cleanedData['Credit_Score']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("X_train")
    st.write(X_train)
    st.write("y_test")
    st.write(y_train)
    st.write("X_test")
    st.write(X_test)
    st.write("y_test")
    st.write(y_test)
    

