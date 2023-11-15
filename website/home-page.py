import streamlit as st
import pandas as pd
from io import StringIO

import numpy as np
import seaborn as sns
#import matplotlib.pyplot as plt

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
    print(DF)
    #sns.pairplot(DF)
