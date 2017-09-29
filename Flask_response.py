# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:34:43 2017

@author: lakshya.khanna
"""

import json
import requests
import os
import pandas as pd

os.chdir('E:\POC\Hospital Readmission POC\Code\Flask')

"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

"""Reading test batch
"""
df = pd.read_csv('rf_v16_test_flask.csv',nrows=2)
df = df.head()

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')

"""POST <url>/predict
"""
resp = requests.post("http://172.16.0.237:8000/predict", \
                    data = json.dumps(data),\
                    headers= header)

resp.json()