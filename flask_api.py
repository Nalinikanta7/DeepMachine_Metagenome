# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 02:20:31 2020

@author: Nalinikanta Choudhury
"""
# Import Libraries
from flask import Flask, render_template, request, session, redirect
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug import secure_filename
# from flask_mail import Mail
import json
import os
import math
from datetime import datetime


app=Flask(__name__)
Swagger(app)

# Import Pickle File
pickle_in_svm = open("model_svm.pkl","rb")
classifier_svm=pickle.load(pickle_in_svm)

pickle_in_rf = open("model_RF.pkl","rb")
classifier_rf= pickle.load(pickle_in_rf)

@app.route('/')
def welcome():
    return "The Deep Machine in river metagenome"

@app.route('/predict_svm',methods=["Get"])
def predict_svm(C,kernel):

    """Let's SVM machine learning 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: C               
        in: query
        type: number
        required: true

      - name: kernel
        in: query
        type: linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        required: true
      
    responses:
        200:
            description: The output values
        
    """

    C       = request.args.get("C")
    kernel  = request.args.get("kernel")

    pred_svm = classifier_svm.predict_svm([[C,kernel]])
    print(pred_svm)
    return "The result is"+str(pred_svm)

@app.route('/predict_rf',methods=["Get"])
def predict_rf(n_estimators,criterion):
    """Random Forest machine learning 
    This is using docstrings for specifications.
    ---
    parameters:  
        - name: n_estimators             
        in: query
        type: int
        required: true

        - name: criterion
        in: query
        type:  ‘gini’, ‘entropy’
        required: true

        - name: max_depth
        in: query
        type: int
        required: true

    responses:
        200:
            description: The output values
        
    """   
    n_estimators  = request.args.get("n_estimators")
    criterion     = request.args.get("criterion")
    # max_depth     = request.args.get("max_depth")

    pred_rf =classifier_rf.predict_rf([[n_estimators,criterion,max_depth]])
    print(pred_rf)
    return "The result is"+str(pred_rf)




@app.route('/predict_svm',methods=["POST"])
def predict_svm_file():
    """Let's do machine Learning SVM
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test_svm=pd.read_csv(request.files.get("file"))
    print(df_test_svm.head())
    pred_svm=classifier_svm.predict_svm(df_test_svm)
    
    return str(list(pred_svm))

@app.route('/predict_rf',methods=["POST"])
def predict_rf_file():
    """Let's do machine Learning Random Forest
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test_rf=pd.read_csv(request.files.get("file"))
    print(df_test_rf.head())
    pred_rf=classifier_rf.predict_rf(df_test_rf)
    
    return str(list(pred_rf))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)