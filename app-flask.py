import numpy as np
from flask import Flask, request, jsonify, Blueprint
import pickle
import os
from src.train import train_model

app = Flask(__name__)


@app.route("/")
def verifica_api_online():
  return "API ONLINE v1.0st", 200

@app.route("/train/", methods=['POST'])
def train():
  train_model()
  return "API ONLINE v1.0st", 200

#
if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port, debug=True)