import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def prepare_data():
    

    data = pd.read_csv('../data/breast cancer data.csv')
    data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    data.dropna(inplace=True)

    X = data.drop(columns=['diagnosis'])
    le = LabelEncoder()
    y = le.fit_transform(data['diagnosis'])

    #Split in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
   

    pickle.dump(le, open('../encoders/label_encoder.sav', 'wb'))
    pickle.dump(scaler, open('../scalers/scaler.sav', 'wb'))

    return X_scaled, y, X_train_scaled, X_test_scaled, y_train, y_test