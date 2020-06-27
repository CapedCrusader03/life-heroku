#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:19:49 2020

@author: kshitij
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('draft.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = draft.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Life expectancy should be : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)