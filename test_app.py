import pandas as pd
import streamlit as st
import plotly.express as px

st.title("Datalab Sample Version")

c1, c2, c3, c4 = st.columns((2, 1, 1, 1))

data1 = st.file_uploader("Upload File 1 Here", type={"csv"})
data2 = st.file_uploader("Upload File 2 Here", type={"csv"})

try:
    
    if data1 is not None:
        data_df1 = pd.read_csv(data1)
    st.write("First 5 Rows of File 1")
    st.write(pd.DataFrame(data_df1).head())
    st.write("Last 5 Rows of File 1")
    st.write(pd.DataFrame(data_df1).tail())
    
    if data2 is not None:
        data_df2 = pd.read_csv(data2)
    st.write("First 5 Rows of File 2")
    st.write(pd.DataFrame(data_df2).head())
    st.write("Last 5 Rows of File 2")
    st.write(pd.DataFrame(data_df2).tail())
    
    st.write("Plot of File 1")
    fig1 = px.line(data_df1, x='date', y='median_price_12')
    st.write(fig1)
    
    st.write("Plot of File 2")
    fig2 = px.line(data_df2, x='date', y='median_price_12')
    st.write(fig2)
    
    corr = data_df1.corrwith(data_df2)['median_price_12']
    st.write("Correlation of File 1 and 2")
    st.write(corr)

except NameError:
    pass