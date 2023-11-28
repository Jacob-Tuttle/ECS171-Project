from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import streamlit as st
sns.set(color_codes=True)
import numpy as np
from sklearn.metrics import classification_report, mean_squared_error
from sklearn import preprocessing

class LinearSVM:
    def __init__(self, learning_rate=0.001, epochs=10000, C=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C
        self.weights = None
        self.bias = None

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        min = float('inf')
        max = float('inf')
        for epoch in range(self.epochs):
            # print(self.weights)
            for i in range(num_samples):
                pred = np.dot(X[i], self.weights) - self.bias
                condition = y[i] * pred > 0
                if not condition:
                    self.weights -= self.learning_rate * (2 * self.C * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) - self.bias)

def linearReport(df):
    X, y = df.drop(columns=['Credit_Score']), df['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    svc_li = SVC(kernel='linear')

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    Z_svm_train = scaler.transform(X_train)
    Z_svm_test = scaler.transform(X_test)

    svc_li.fit(Z_svm_train, np.asarray(y_train))
    predictions = svc_li.predict(Z_svm_test)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    # Print MSE using Streamlit
    st.write("MSE: ", mse)
    
    return classification_report(y_test, predictions)

def nonLinearReport(df):
    X, y = df.drop(columns=['Credit_Score']), df['Credit_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # w/ rbf kernel
    svc_rbf = SVC(kernel='rbf')

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    Z_svm_train = scaler.transform(X_train)
    Z_svm_test = scaler.transform(X_test)

    svc_rbf.fit(Z_svm_train, np.asarray(y_train))

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    # Print MSE using Streamlit
    st.write("MSE: ", mse)
    
    return classification_report(y_test, svc_rbf.predict(Z_svm_test))
