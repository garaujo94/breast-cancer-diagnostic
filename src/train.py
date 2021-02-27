import numpy as np 
import pandas as pd
from src.data import prepare_data
from src.train_diagnosis import train_breast_cancer
import streamlit as st



def train_model(is_streamlit = False):
    #Manda pré-processar as informações
    print('Preparing data...')
    if is_streamlit:
        st.write('Preparing data...')
    X_full, y_full, X_train, y_train, X_test, y_test = prepare_data()
    if is_streamlit:
        st.write('Preparing data... OK')
    print('Preparing data... OK')
    #Manda treinar
    print('Trainning data...')
    if is_streamlit:
        st.write('Trainning data...')
    response = train_breast_cancer(X_full, y_full, X_train, y_train, X_test, y_test, is_streamlit)
    print('Trainning data... OK')
    if is_streamlit:
        st.write('Trainning data... OK')
    #Retorna sucesso
    print(response)

