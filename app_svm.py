# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 02:20:31 2020

@author: Nalinikanta Choudhury
"""

# Import Libraries
import numpy as np
import pickle
import pandas as pd
# from flasgger import Swagger
import streamlit as st
from PIL import Image

# Import Pickle File
pickle_in = open("model_svm.pkl","rb")
classifier=pickle.load(pickle_in)

page_bg_img = '''
<style>
    body {
    background-image: url("https://images.unsplash.com/photo-1528458909336-e7a0adfed0a5");
    background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# @app.route('/')
def welcome():
    return "Welcome to Deep Machine in River Metagenomes"

# @app.route('/predict',methods=["Get"])
def predict_river_metagenome(C,kernel,degree,gamma):

  """Let's Authenticate the Banks Note 
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
    - name: degree
      in: query
      type: number
      required: true
    - name: gamma
      in: query
      type: ‘scale’, ‘auto’
      required: true
  responses:
      200:
          description: The output values
      
  """
   
  prediction = classifier.predict([[C, kernel, degree, gamma]])
  print(prediction)
  return prediction

def main():
    st.title("The Deep Machine in river Metagenome")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">The Deep Machine in river Metagenome ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    C      = st.text_input("C","Type Here")
    kernel = st.text_input("kernel","Type Here")
    degree = st.text_input("degree","Type Here")
    gamma  = st.text_input("gamma","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_river_metagenome(C, kernel, degree, gamma)
    st.success('The output is {}'.format(result))
    if st.button("# Contact Us"):
        st.write("""
        *Dr A.R. Rao, ADG, ICAR*
        
        *Nalinikanta Choudhury, PhD Bioinformatics*
        """)

if __name__=='__main__':
    main()