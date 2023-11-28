import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import re
from pathlib import Path

def removeOutliers(a):
    df = a
    # Drop the column which is out of model scope
    d_col = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation', 'Type_of_Loan', 'Credit_Mix',
             'Payment_of_Min_Amount', 'Payment_Behaviour',
             'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
             'Num_of_Loan', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
             'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    drop_df = df.drop(d_col, axis=1).copy()
    drop_df.isnull().sum()
    drop_na = drop_df.dropna().copy()
    # Revise the incorrect data whole table
    sym = "\\`*_{}[]()>#@+!$:;"
    col_int = ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt']
    col_str = ['Credit_Score', 'Credit_History_Age']
    for i in col_int:
        for c in sym:
            drop_na[i] = drop_na[i].astype(str).str.replace(c, '')
    for i in col_str:
        for c in sym:
            drop_na[i] = drop_na[i].replace(c, '')

    # Transform the information to the value
    drop_na['Credit_History_Age'] = drop_na['Credit_History_Age'].astype(str).str.replace(' Years and ', '.')
    drop_na['Credit_History_Age'] = drop_na['Credit_History_Age'].astype(str).str.replace('Months', '')

    # Transform the object data the be float data type
    col_int2 = ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Outstanding_Debt', 'Credit_History_Age']
    for i in col_int2:
        drop_na[i] = drop_na[i].astype(float)

    df_cleaned = drop_na

    Q1 = df_cleaned.Credit_History_Age.quantile(0.25)
    Q3 = df_cleaned.Credit_History_Age.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Credit_History_Age'] > (Q3 + 1.5 * IQR)].index)
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Credit_History_Age'] < (Q1 - 1.5 * IQR)].index)

    Q1 = df_cleaned.Credit_Utilization_Ratio.quantile(0.25)
    Q3 = df_cleaned.Credit_Utilization_Ratio.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Credit_Utilization_Ratio'] > (Q3 + 1.5 * IQR)].index)
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Credit_Utilization_Ratio'] < (Q1 - 1.5 * IQR)].index)

    Q1 = df_cleaned.Delay_from_due_date.quantile(0.25)
    Q3 = df_cleaned.Delay_from_due_date.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Delay_from_due_date'] > (Q3 + 1.5 * IQR)].index)
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Delay_from_due_date'] < (Q1 - 1.5 * IQR)].index)

    Q1 = df_cleaned.Outstanding_Debt.quantile(0.25)
    Q3 = df_cleaned.Outstanding_Debt.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Outstanding_Debt'] > (Q3 + 1.5 * IQR)].index)
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Outstanding_Debt'] < (Q1 - 1.5 * IQR)].index)

    Q1 = df_cleaned.Num_of_Delayed_Payment.quantile(0.25)
    Q3 = df_cleaned.Num_of_Delayed_Payment.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Num_of_Delayed_Payment'] > (Q3 + 1.5 * IQR)].index)
    df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['Num_of_Delayed_Payment'] < (Q1 - 1.5 * IQR)].index)

    return df_cleaned