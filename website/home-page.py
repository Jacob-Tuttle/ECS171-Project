import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from outliers import removeOutliers
import logistic
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
    data['Credit_Score'] = data['Credit_Score'].replace(creditScoreMap)

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

    st.write("Logistic")
    st.write(logistic.report(cleanedData))
    st.write("Linear SVM")
    st.write(svm.linearReport(cleanedData))
    st.write("Non-Linear SVM")
    st.write(svm.nonLinearReport(cleanedData))

