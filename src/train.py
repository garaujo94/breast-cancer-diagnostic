import numpy as np 
import pandas as pd
from data import prepare_data
from train_diagnosis import train_breast_cancer



def train_model():
    #Manda pré-processar as informações
    X_full, y_full, X_train, y_train, X_test, y_test = prepare_data()
    #Manda treinar
    response = train_breast_cancer(X_full, y_full, X_train, y_train, X_test, y_test)
    #Retorna sucesso
    print(response)

train_model()