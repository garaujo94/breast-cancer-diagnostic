import numpy as np 
import pandas as pd
from src.data import prepare_data
from src.train_diagnosis import train_breast_cancer
import streamlit as st
import pickle


def predict_from_csv_file(X_to_predict):

    model = pickle.load(open('models/trained_model.sav', 'rb'))
    predict = model.predict(X_to_predict)[0]

    print(predict)

    return predict

def predict_from_uri():
    return None