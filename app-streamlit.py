import streamlit as st
from src.train import train_model

st.title('''
Breast Cancer Diagnosis with Python
''')
st.write('''
According to the Brazilian National Cancer Institute (INCA), breast cancer is a disease caused by the disordered multiplication of breast cells. 
There are different types of breast cancer, so each case can evolve differently. 
Despite representing around 1% of cases of the disease, even men can suffer from illness. 
Also according to the INCA, there are an estimated 66,280 new cases for the year 2020.

This is a proof of concept using streamlit for a breast cancer diagnostic model, since one of the fundamental parts of a project is to allow the end user to have access to the product and be able to use it
''')


if st.button('Train Model!'):
    with st.spinner('Starting trainning the Model!'):
        train_model(True)
    st.success('Model Trained and Saved!')
    