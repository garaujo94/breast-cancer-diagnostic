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

With the current pandemic of COVID-19, much has been said about using Artificial Intelligence to fight the virus. 
These are initiatives that range from combining substances in order to try to find an effective remedy through super computers, 
to Neural Networks capable of providing a diagnosis about COVID-19 based on radiographs of the patient's thoracic region. 
It turns out that attempts to assist doctors' decisions in their professions did not begin because of the new corona virus.

Today, I bring you a database on breast cancer patients and their tumors and our goal will be to create a model that will try to understand 
whether the tumor in question is benign or malignant. It is worth remembering that the goal is not to replace the doctor with a computer program, 
but to try to help medical decisions so that one can help save lives.
''')

if st.button('Train Model!'):
    st.write('Starting trainning the Model!')
    train_model(True)
    st.write('Model Trained and Saved!')
    