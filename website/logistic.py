# Exploratory data analysis
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import streamlit
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


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

    # Evaluate the model
    # print("Accuracy:", accuracy_score(y_test, predictions))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    # print("Classification Report:\n", classification_report(y_test, predictions))
    total = 0
    for pred in predictions:
        total += (y_test - pred)**2

    st.write("MSE: ", total/len(predictions)
    return classification_report(y_test, predictions)
