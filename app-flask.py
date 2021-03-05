import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
import pickle
import os
from src.train import train_model
from src.predict import predict_from_trained_model
from src.data import load_data_to_predict, prepare_data_to_predict_from_uri

app = Flask(__name__)


@app.route("/")
def verifica_api_online():
    return "API ONLINE v1.0st", 200

@app.route("/train/", methods=['POST'])
def train(event=None, _=None):
    train_model()
    return "API ONLINE v1.0st", 200

@app.route("/predict_from_csv/", methods=['POST'])
def predict_from_csv(event=None, _=None):
    body = request.args
    file_id = body['file_id']
    
    data = load_data_to_predict(file_id)
    predict = predict_from_trained_model(data)
    return f"API ONLINE v1.0st - {predict}", 200

@app.route('/predict/', methods=['POST'])
def predict(event=None, _=None):
    body = request.args
    body = body.to_dict()
    
    data = prepare_data_to_predict_from_uri(body)
    predict = predict_from_trained_model(data)

    return f"API ONLINE v1.0st - {predict}", 200
  


#
if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port, debug=True)