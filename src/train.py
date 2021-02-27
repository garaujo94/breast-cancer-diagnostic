import numpy as np 
import pandas as pd
from src.data import prepare_data
from src.train_diagnosis import train_breast_cancer



def train_model():
    #Manda pré-processar as informações
    print('Preparing data...')
    X_full, y_full, X_train, y_train, X_test, y_test = prepare_data()
    print('Preparing data... OK')
    #Manda treinar
    print('Trainning data...')
    response = train_breast_cancer(X_full, y_full, X_train, y_train, X_test, y_test)
    print('Trainning data... OK')
    #Retorna sucesso
    print(response)

