import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def pca_formula(df):

    numerical_cl = []
    categorical_cl = []

    for i in df.columns:
        if df[i].dtype == np.dtype("float64") or df[i].dtype == np.dtype("int64"):
            numerical_cl.append(df[i])
        else:
            categorical_cl.append(df[i])

    numerical_data = pd.concat(numerical_cl, axis=1)

    categorical_data = pd.concat(categorical_cl, axis=1)

    numerical_data = numerical_data.apply(lambda x: x.fillna(np.mean(x)))

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(numerical_data)

    pca = PCA()

    pca_data = pca.fit_transform(scaled_values)

    pca_data = pd.DataFrame(pca_data)

    new_columns_name = ["PCA_"+ str(i) for i in range(1, len(pca_data.columns)+1)]

    column_map = dict(zip(list(pca_data.columns), new_columns_name))

    pca_data.rename(columns=column_map, inplace=True)

    output = pd.concat([df, pca_data],axis=1)

    return output, categorical_cl, new_columns_name







st.set_page_config(layout="wide")
scatter_column, settings_column = st.columns((4, 1))

scatter_column.title("Multi-Dimensional Analysis")

settings_column.title("Settings")

uploaded_file = settings_column.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    pca_data, cat_cols, pca_cols = pca_formula(df)

    categorical_variable = settings_column.selectbox("Variable Select", options = cat_cols)
    categorical_variable_2 = settings_column.selectbox("Second Variable Select", options = cat_cols)

    pca_1 = settings_column.selectbox("First Principle Component", options=pca_cols, index=0)
    pca_cols.remove(pca_1)
    pca_2 = settings_column.selectbox("Second Principle Component", options=pca_cols)

    scatter_column.plotly_chart(px.scatter(data_frame=pca_data, x=pca_1, y=pca_2, color=categorical_variable, template="simple_white", height=900, hover_data = [categorical_variable_2]), use_container_width=True)
else:
    scatter_column.header("No file uploaded")

