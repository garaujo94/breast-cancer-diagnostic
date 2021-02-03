import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle


def train_breast_cancer(X, y, X_train, y_train, X_test, y_test):
    
    model = KNeighborsClassifier()
    model.fit(X_train,y_train)

    #Printing metrics
    pred_train = model.predict(X_train)
    print(classification_report(y_train, pred_train))

    pred_test = model.predict(X_test)
    print(classification_report(y_test, pred_test))

    final_model = KNeighborsClassifier()
    final_model.fit(X, y)

    pickle.dump(final_model, open('../models/trained_model.sav', 'wb'))
    print('model saved at models folds')

    return '200'