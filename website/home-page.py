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

    cleanedData = removeOutliers(data)

    # Map credit scores to number
    creditScoreMap = {'Poor': 1, 'Standard': 2, 'Good': 3}
    cleanedData['Credit_Score'] = cleanedData['Credit_Score'].replace(creditScoreMap)

    # Separate features (X) and target variable (y)
    X = cleanedData.drop(columns=['Credit_Score'])
    y = cleanedData['Credit_Score']
    
    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    selected_models = st.sidebar.multiselect("Select Models", ["Logistic Regression", "SVM Linear Regression", "SVM Non-Linear Regression"])
    
    # Button to trigger the classification report
    if st.sidebar.button("Run"):
        if not selected_models:
            st.warning("Please select at least one model.")
        else:
            for model_name in selected_models:
                if model_name == "Logistic Regression":
                    report = logistic.report(cleanedData)
                elif model_name == "SVM Linear Regression":
                    report = svm.linearReport(cleanedData)
                elif model_name == "SVM Non-Linear Regression":\
                    report = svm.nonLinearReport(cleanedData)
                    
                st.text(f"Classification Report for {model}:\n{report}")

