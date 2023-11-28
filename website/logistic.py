# Exploratory data analysis
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression

def plot_predictions_vs_actual(y_test, predictions):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs Actual Values")
    st.pyplot()

def report(data):
    X = data.drop(columns=['Credit_Score'])
    y = data['Credit_Score']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a multiclass logistic regression model
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    # Fit the model on the training data
    logreg.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = logreg.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    # Print MSE using Streamlit
    st.write("MSE: ", mse)
    
    plot_predictions_vs_actual(y_test, predictions)
    return classification_report(y_test, predictions)
