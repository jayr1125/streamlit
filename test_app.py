import pandas as pd
import streamlit as st

st.title("Hi Rod! Welcome to Streamlit")

st.write("My first dataframe")

st.write(
    pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })
)
