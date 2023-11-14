import streamlit as st
import pandas as pd
from io import StringIO

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
    dataframe = pd.read_csv(uploaded_file)
    DF = pd.DataFrame(dataframe)

drop = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age', 'Payment_of_Min_Amount', 'Payment_Behaviour',
        'Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',  'Num_Credit_Inquiries',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Num_of_Delayed_Payment']

# Columns to clean
clean = ['Delay_from_due_date', 'Monthly_Inhand_Salary', 'Monthly_Balance', 'Changed_Credit_Limit','Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age']

# Clean columns
for item in clean:
    train[item] = pd.to_numeric(train[item], errors='coerce')

# Map credit scores to number
creditScoreMap = {'Poor': 1, 'Standard': 2, 'Good': 3}
train['Credit_Score'] = train['Credit_Score'].replace(creditScoreMap)


data = train.copy().drop(columns=drop)

# Separate features (X) and target variable (y)
X = data.drop(columns=['Credit_Score'])
y = data['Credit_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values, Logistic regression can not take in NaN values
imputer = SimpleImputer(strategy='mean')  # You can use 'median' or 'most_frequent' as well
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create a multiclass logistic regression model
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Fit the model on the training data
logreg.fit(X_train_imputed, y_train)

# Make predictions on the test data
predictions = logreg.predict(X_test_imputed)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(20, 20))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap='RdBu')
plt.show()
