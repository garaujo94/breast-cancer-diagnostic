import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle
from pathlib import Path
import streamlit as st

path = Path(__file__).parent

def train_breast_cancer(X, y, X_train, y_train, X_test, y_test, is_streamlit=False):
    try:

        model = KNeighborsClassifier()
        model.fit(X_train,y_train)

        #Printing metrics
        pred_train = model.predict(X_train)
        if is_streamlit:
            st.write('TRAIN REPORT')

            st.table(pd.DataFrame(classification_report(y_train, pred_train, output_dict=True)).T[['precision', 'recall','f1-score']].iloc[[0,1]])
            
            st.write('Accuracy:')
            st.write(classification_report(y_train, pred_train, output_dict=True)['accuracy'])
        
        print('===========================================')
        print('TRAIN REPORT')
        print(classification_report(y_train, pred_train))
        print('===========================================')
        pred_test = model.predict(X_test)
        if is_streamlit:
            st.write('TEST REPORT')
            st.table(pd.DataFrame(classification_report(y_test, pred_test, output_dict=True)).T[['precision', 'recall','f1-score']].iloc[[0,1]])
            st.write('Accuracy:')
            st.write(classification_report(y_test, pred_test, output_dict=True)['accuracy'])
        else:
            print('TEST REPORT')
            print(classification_report(y_test, pred_test))
            print('===========================================')

        final_model = KNeighborsClassifier()
        final_model.fit(X, y)

        file_path = (path / "../models/trained_model.sav").resolve()
        pickle.dump(final_model, open(file_path, 'wb'))
        print('model saved at models folds')

        response = '200'    
    except:
        response = 'Error at Training Step'
        
    return response