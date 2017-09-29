# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:19:48 2017

@author: lakshya.khanna
"""

from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import os
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST','GET'])
def predict():
     json_ = request.json
     data = json.loads(json_)
     query_df = pd.DataFrame(data)
     prediction = dt_estimator.predict(query_df)
     d = {}
     for x, i in enumerate(list(prediction)):
         d[str(x)] = int(i)
     return json.dumps(d)

 
MODEL_DIR = 'E:\POC\Hospital Readmission POC\Code\Best Models'
MODEL_FILE = 'RF_65.95_ROC-AUC_v16.pkl'
if __name__ == '__main__':
     os.chdir(MODEL_DIR)
     dt_estimator = joblib.load(MODEL_FILE)
     app.run(host = '172.16.0.237',port=8000)