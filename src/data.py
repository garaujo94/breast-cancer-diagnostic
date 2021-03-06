import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from pathlib import Path

path = Path(__file__).parent


def prepare_data():
    

    data = pd.read_csv('data/breast_cancer_data.csv')
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
   
    file_path = (path / "../encoders/label_encoder.sav").resolve()
    pickle.dump(le, open(file_path, 'wb'))
    file_path = (path / "../scalers/scaler.sav").resolve()
    pickle.dump(scaler, open(file_path, 'wb'))

    return X_scaled, y, X_train_scaled, y_train, X_test_scaled, y_test


def load_data_to_predict(id):

    data = pd.read_csv(f'data/to_predict/{id}.csv')
    data.drop(columns=['id'], inplace=True)
    data.dropna(inplace=True)

    file_path = (path / "../scalers/scaler.sav").resolve()
    scaler = pickle.load(open(file_path, 'rb'))

    data = scaler.transform(data) 

    return data

def prepare_data_to_predict_from_uri(data):

    data = pd.DataFrame(data=data, index = [0])
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    data.dropna(inplace=True)

    file_path = (path / "../scalers/scaler.sav").resolve()
    scaler = pickle.load(open(file_path, 'rb'))

    data = scaler.transform(data) 

    return data

def deconding_prediction(y):

    file_path = (path / "../encoders/label_encoder.sav").resolve()
    le = pickle.load(open(file_path, 'rb'))

    decoded = le.inverse_transform(np.array(y))[0]

    mapper = {
        'B': 'Benign',
        'M': 'Malignant'
    }

    r = mapper[decoded]

    return r