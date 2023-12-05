import joblib
import pandas as pd

logreg = joblib.load("logistic_model.pkl")

svmLinear = joblib.load("svm_linear_model.pkl")

svmRBF = joblib.load("svm_rbf_model.pkl")
input = {
    'Delay_from_due_date': 1,
    'Num_of_Delayed_Payment': 1,
    'Outstanding_Debt': 1,
    'Credit_Utilization_Ratio': 1,
    'Credit_History_Age': 1,
}

df = pd.DataFrame([input])
prediction1 = logreg.predict(df)
prediction2 = svmLinear.predict(df)
prediction3 = svmRBF.predict(df)

print(f"Logistic Predicted Credit Score: {prediction1[0]}")
print(f"SVM Linear Predicted Credit Score: {prediction2[0]}")
print(f"SVM RBF Predicted Credit Score: {prediction3[0]}")
