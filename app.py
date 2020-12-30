# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 02:20:31 2020

@author: Nalinikanta Choudhury
"""

# Import Libraries
import numpy as np
import pickle
import pandas as pd
import os
# from flasgger import Swagger
import streamlit as st
from PIL import Image
import time
import matplotlib.pyplot as plt
# import json
# from classify import predict


# Import Pickle File
# pickle_in_svm = open("model_svm.pkl","rb")
# classifier_svm=pickle.load(pickle_in_svm)

# pickle_in_rf = open("model_RF.pkl","rb")
# classifier_rf=pickle.load(pickle_in_rf)

# pickle_in_lstm = open("model_LSTM.pkl","rb")
# classifier_lstm=pickle.load(pickle_in_lstm)

page_bg_img = '''
<style>
    body{
    background-image: url("https://images.unsplash.com/photo-1528458909336-e7a0adfed0a5");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# <body>
#     <img src="Header.jpg" style=" margin:0px; width:868px;height:134px; padding:0px; border:0px">
# </body>


# st.image(load_image(Header.jpg))


# # Upload file
# st.sidebar.title("Upload your file")

# uploaded_file = st.sidebar.file_uploader("Choose a feature file...", type="csv")
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     # pd.read_csv(data, caption = 'uploaded file.',use_column_width =True)
#     st.write("")
#     st.write("Classifying....")
#     label = predict(uploaded_file)
#     st.write('%s (%.2f%%)' % (label[1], label[2]*100))
# else:
#     print("Upload toh kar MC...")



####### Sticky Header
# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded

# header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
#     img_to_bytes("Header.jpg")
# )
# st.markdown(header_html, unsafe_allow_html=True)


###### Prediction function
def predict_svm(C,kernel, dataset):
    """SVM machine learning 
    This is using docstrings for specifications.
    ---
    parameters:  
        - name: C               
        in: query
        type: int
        required: true
        - name: kernel
        in: query
        type: 'linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        required: true
    responses:
        200:
            description: The output values
        
    """
    # C = st.sidebar.slider("C ==> Regularization parameter",1, 100, 1)
    # kernel = st.sidebar.radio("Kernel ==> Specifies the kernel type", ["rbf","linear","poly","sigmoid"])
    if (C == 1):
        if(kernel == 'rbf'):
            pickle_in_svm_1 = open("model_svm_1.pkl","rb")
            classifier_svm_1 = pickle.load(pickle_in_svm_1)
            pred_svm_1 = classifier_svm_1.predict(dataset)
            print(pred_svm_1)
            return pred_svm_1
    

    if (C == 10):
        if (kernel == 'rbf'):
            pickle_in_svm_2 = open("model_svm_2.pkl","rb")
            classifier_svm_2 = pickle.load(pickle_in_svm_2)

            pred_svm_2 = classifier_svm_2.predict(dataset)
            print(pred_svm_2)
            
            return pred_svm_2

    if (C == 1):
        if(kernel =='linear'):
            pickle_in_svm_3 = open("model_svm_3.pkl","rb")
            classifier_svm_3 = pickle.load(pickle_in_svm_3)

            pred_svm_3 = classifier_svm_3.predict(dataset)
            print(pred_svm_3)
            
            return pred_svm_3

    if (C == 10):
        if(kernel =='linear'):
            pickle_in_svm_4 = open("model_svm_4.pkl","rb")
            classifier_svm_4 = pickle.load(pickle_in_svm_4)

            pred_svm_4 = classifier_svm_4.predict(dataset)
            print(pred_svm_4)
            
            return pred_svm_4

    if (C == 1):
        if(kernel =='poly'):
            pickle_in_svm_5 = open("model_svm_5.pkl","rb")
            classifier_svm_5 = pickle.load(pickle_in_svm_5)

            pred_svm_5 = classifier_svm_5.predict(dataset)
            print(pred_svm_5)
            
            return pred_svm_5

    if (C == 10):
        if(kernel =='poly'):
            pickle_in_svm_6 = open("model_svm_6.pkl","rb")
            classifier_svm_6 = pickle.load(pickle_in_svm_6)

            pred_svm_6 = classifier_svm_6.predict(dataset)
            print(pred_svm_6)
            
            return pred_svm_6

    if (C == 1):
        if(kernel =='sigmoid'):
            pickle_in_svm_7 = open("model_svm_7.pkl","rb")
            classifier_svm_7 = pickle.load(pickle_in_svm_7)

            pred_svm_7 = classifier_svm_7.predict(dataset)
            print(pred_svm_7)
            
            return pred_svm_7

    if (C == 10):
        if(kernel =='sigmoid'):
            pickle_in_svm_8 = open("model_svm_8.pkl","rb")
            classifier_svm_8 = pickle.load(pickle_in_svm_8)

            pred_svm_8 = classifier_svm_8.predict(dataset)
            print(pred_svm_8)
            
            return pred_svm_8


def predict_rf(n_estimators,criterion, dataset):
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
    # n_estimators = st.sidebar.slider("n_estimators",10, 100, 16)
    # criterion = st.sidebar.radio("criterion", ["gini", "entropy"])
    # max_depth  = st.sidebar.slider("max_depth", 0, 50, 0)    
    # pred_rf = classifier_rf.predict(n_estimators, criterion, max_depth)
    # print(pred_rf)
    
    # return pred_rf

    if (n_estimators == 10):
        if(kernel =='gini'):
            pickle_in_rf_1 = open("model_rf_1.pkl","rb")
            classifier_rf_1 = pickle.load(pickle_in_rf_1)

            pred_rf_1 = classifier_rf_1.predict(dataset)
            print(pred_rf_1)
            
            return pred_rf_1

    if (n_estimators == 16):
        if(kernel =='gini'):
            pickle_in_rf_2 = open("model_rf_2.pkl","rb")
            classifier_rf_2 = pickle.load(pickle_in_rf_2)

            pred_rf_2 = classifier_rf_2.predict(dataset)
            print(pred_rf_2)
            
            return pred_rf_2

    if (n_estimators == 10):
        if(kernel =='entropy'):
            pickle_in_rf_3 = open("model_rf_3.pkl","rb")
            classifier_rf_3= pickle.load(pickle_in_rf_3)

            pred_rf_3 = classifier_rf_3.predict(dataset)
            print(pred_rf_3)
            
            return pred_rf_3

    if (n_estimators == 16):
        if(kernel =='entropy'):
            pickle_in_rf_4 = open("model_rf_4.pkl","rb")
            classifier_rf_4 = pickle.load(pickle_in_rf_4)

            pred_rf_4 = classifier_rf_4.predict(dataset)
            print(pred_rf_4)
            
            return pred_rf_4

######### Main Function
def main():
    st.sidebar.title("Upload your file:")
    dataset = st.sidebar.file_uploader("Choose a feature file...", type="csv")
    # if st.sidebar.button("Predict the microorganism with SVM"):

    #     if dataset is not None:
    #         file_details = {"Filename":dataset.name,"FileType":dataset.type,"FileSize":dataset.size}
    #         data = pd.read_csv(dataset)
    #         st.dataframe(data)


        # pd.read_csv(data, caption = 'uploaded file.',use_column_width =True)
        # st.write("")
        # st.write("Classifying....")
        # label = predict(uploaded_file)
        # st.write('%s (%.2f%%)' % (label[1], label[2]*100))


    # st.title("The Deep Machine in river Metagenome")
    html_temp = """
    <div style="background:#025246o;padding:10px">
    <h2 style="color:black;text-align:center; background color: #b5bd40; font-size: 2.75rem; "><b>The Deep Machine in river Metagenome </b></h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    ######### Choose Classifier
    st.sidebar.title("Select Classifier:")
    # classifier_name = st.sidebar.selectbox ("Select Classifiers",("SVM","Random Forest","LSTM"))
    classifier_name = st.sidebar.selectbox("Select Classifiers", ("SVM","Random Forest"))
    st.write("Your selected classifier is ",classifier_name)

    ######### Choose Parameters
    st.sidebar.title("Choose the Parameters:")
    if classifier_name == "SVM":
        C = st.sidebar.radio("C ==> Regularization parameter",["1", "10"])
        kernel = st.sidebar.radio("Kernel ==> Specifies the kernel type", ["rbf","linear","poly","sigmoid"])
    else:
        n_estimators = st.sidebar.radio("n_estimators",["10", "16"])
        criterion = st.sidebar.radio("criterion", ["gini", "entropy"])
        # max_depth  = st.sidebar.slider("max_depth", 0, 50, 0)  
    # st.write("Your selected paramters are ",paramters_name)

    
    safe_html ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> The Output is as following: </h2>
        </div>
        """
    st.markdown(safe_html, unsafe_allow_html=True)

    ##### Data and Classifier integration   

    if st.sidebar.button("Predict the microorganism"):
        if classifier_name == "SVM":
            if dataset is not None:
                data = pd.read_csv(dataset)
            print(kernel)
            print(C)
            output = predict_svm(C,kernel, data)
            print(output)
            st.success('The predicted microorganism is a {}'.format(output))

            if output == 1:
                st.markdown(Bacteria, safe_html,unsafe_allow_html=True)
            elif output == 0:
                st.markdown(Fungi, warn_html,unsafe_allow_html=True)
            else: 
                st.markdown("unknown")

        elif classifier_name == "Random Forest":
            if dataset is not None:
                data = pd.read_csv(dataset)
            print(n_estimators)
            print(criterion)
            # st.sidebar.button("Predict the microorganism with RF")
            output = predict_rf(n_estimators,criterion,data)
            st.success('The predicted microorganism is a {}'.format(output))

            if output == 1:
                st.markdown(Bacteria, safe_html,unsafe_allow_html=True)
            elif output == 0:
                st.markdown(Fungi, warn_html,unsafe_allow_html=True)
            else: 
                st.markdown("unknown")
        else:
            print("Better luck Next Time..")
    



if __name__=='__main__':
    main()



    
    # Upload file

# st.sidebar.title("Input file")
# filename = st.sidebar.text_input('Enter a file path:')
# st.sidebar.write("Example file path: /home/cabin/Desktop/Deep machine/Kanpur_extracted_feature_with_Response.csv")
# try:
#     with open(filename) as input:
#         st.text(input.read())
# except FileNotFoundError:
#     st.error('File not found.')

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://github.com/Nalinikanta7/DeepMachine_Metagenome/blob/main/extracted_feature_with_Response.csv')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data
#     data_load_state = st.text('Loading data...')
#     data = load_data(10000)
#     data_load_state.text("Done! (using st.cache)")
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)
#     st.subheader('Number of pickups by hour')
#     hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
#     st.bar_chart(hist_values)


#### Training Set Plotting
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('classifier_name (Training set)')
# plt.xlabel('Scaffolds')
# plt.ylabel('Features')
# plt.legend()
# plt.show()

# ########## Test set Plotting
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('classifier_name (Training set)')
# plt.xlabel('Scaffolds')
# plt.ylabel('Features')
# plt.legend()
# plt.show()


# if st.button("About Us"):
#     st.text("Developed By:")
#     st.text("Nalinikanta Choudhury")
#     st.text("Dr A. R. Rao")


# hide_footer_style = """
#     <style>
#     .reportview-container .main footer {visibility: hidden;}    
#     """
# st.markdown(hide_footer_style, unsafe_allow_html=True)

# st.beta_set_page_config(page_title='Deep_Machine', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
# # favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)






















# 'Starting a long computation...'
# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)
# for i in range(100):
# # Update the progress bar with each iteration.
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i + 1)
#     time.sleep(0.1)
# '...and now we\'re done!'



# def predict_lstm(n_estimators,criterion, max_depth):
#     """ LSTM Deep learning 
#     This is using docstrings for specifications.
#     ---
#     parameters:  
#         - name: n_estimators             
#         in: query
#         type: number
#         required: true

#         - name: criterion
#         in: query
#         type:  ‘gini’, ‘entropy’
#         required: true

#         - name: max_depth
#         in: query
#         type: int
#         required: true

#     responses:
#         200:
#             description: The output values
        
#     """
#     prediction = classifier_lstm.predict(n_estimators, criterion, max_depth)
#     print(prediction)
#     return prediction




# ##### Parameters
#     st.sidebar.title("Choose the Parameters")
#     def add_parameter_ui(clf_name):
#         params = dict()
#         if clf_name == "SVM":
#             C = st.sidebar.slider("C ==> Regularization parameter",1, 100, 1)
#             kernel = st.sidebar.radio("Kernel ==> Specifies the kernel type", ["rbf","linear","poly","sigmoid"])
            
#             for i in range(C, kernel):
#                 params["i"] = i

#         elif clf_name == "Random Forest":
#             n_estimators = st.sidebar.slider("n_estimators",10, 100, 16)
#             criterion = st.sidebar.radio("criterion", ["gini", "entropy"])
#             max_depth  = st.sidebar.slider("max_depth", 0, 50, 0)
            
#             for j in range(n_estimators,criterion, max_depth):
#                 params["j"] = j
#         elif clf_name == "LSTM":
#             activation = st.sidebar.selectbox("Activation Function", ["linear", "sigmoid", "GRU", "BiLSTM", "tanh"])
#             params["activation"] = activation
    
#         return params

#     add_parameter_ui(classifier_name)

# ################################################################################################################################################
# ##### Fitting classifiers using Parameters 

#     def get_classifier(clf_name, params):
#         if clf_name == "SVM":
#             clf = SVC(C = params["C"],
#                     kernel= params["kernel"])

#         elif clf_name == "Random Forest":
#             clf = RandomForestClassifier(n_estimators = params["n_estimators"],
#                                         criterion = params["criterion"],
#                                         max_depth = params["max_depth"], random_state=1234)
#         elif clf_name == "LSTM":
#             clf = LSTM(activation= params["activation"] )
        
#         return clf

#     clf = get_classifier(classifier_name, params)

#     prediction = classifier.predict([[C, kernel, degree, gamma]])
#     print(prediction)
#     return prediction