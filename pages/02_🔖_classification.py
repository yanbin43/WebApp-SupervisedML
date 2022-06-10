import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.title('Under Construction')

#--------------------------------------------------------------------------------------------------------------#

# st.title('Machine Learning: Classification')

# st.sidebar.title('Start here:')
# data_option = st.sidebar.radio(label = 'Select Dataset Type',
#                                options = ['Default', 'User-Defined'])

#--------------------------------------------------------------------------------------------------------------#

# if data_option == 'Default':
    
#     selected_data = st.sidebar.selectbox(label = 'Select a Default Dataset',
#                                          options = ['Iris', 'Wine', 'Breast Cancer'])
    
#     if selected_data == 'Iris':
#         data = datasets.load_iris()
#     elif selected_data == 'Wine':
#         data = datasets.load_wine()
#     else:
#         data = datasets.load_breast_cancer()
        
#     X = pd.DataFrame(data.data, columns = data.feature_names)
#     y = pd.DataFrame(data.target, columns = ['label'])
    
#     st.header(f'Dataset Chosen: {selected_data}')
#     descr = st.expander('About the Dataset')
#     descr.write(data.DESCR)

    
#--------------------------------------------------------------------------------------------------------------#

# else:
    
#     uploaded_data = st.sidebar.file_uploader(label = 'Upload a csv file',
#                                              type = 'csv')
    
#     if uploaded_data == None:
        
#         st.title('Please upload a dataset to continue.')
        
#     else:

#         data = pd.read_csv(uploaded_data)
#         cols = list(data.columns)
        
#         y_name = st.sidebar.selectbox(label = 'Select the target (y)', options = sorted(cols))
#         y = data[y_name]
        
#         cols.remove(y_name)
#         X_names = st.sidebar.multiselect(label = 'Select the features (X)', options = sorted(cols))
#         X = data[X_names]
        
#         st.header(f'Dataset Chosen: {uploaded_data.name}')
        

#--------------------------------------------------------------------------------------------------------------#

# clf = st.sidebar.selectbox(label = 'Select a Classifier', 
#                            options = ['k-Nearest Neighbours', 'Support Vector Machine', 'Logistic Regression',
#                                       'Naive Bayes', 'Decision Tree', 'Random Forest'])

# test_size = st.sidebar.number_input(label = 'Set a Test Size', min_value = 0.05, max_value = 0.40, value = 0.20, step = 0.05)

# random_state = st.sidebar.number_input(label = 'Set a Random State', min_value = 0, max_value = 10000, value = 100, step = 1)


#--------------------------------------------------------------------------------------------------------------#

# try:
    
#     st.header('Dataset Overview')
#     data_full = pd.concat([X, y], axis = 1)
#     st.write(data_full.describe())


#     #--------------------------------------------------------------------------------------------------------------#

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
#     if clf == 'k-Nearest Neighbours':
#         model = 
      
    
#     model = LinearRegression()
#     model.fit(X, y)
#     y_pred = model.predict(X_test)
