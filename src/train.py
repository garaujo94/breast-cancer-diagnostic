import numpy as np 
import pandas as pd
from utils.data import prepare_data




def train_model():
    #Manda pré-processar as informações
    X_full, y_full, X_train, X_test, y_train, y_test = prepare_data()
    #Manda treinar

    #Retorna sucesso