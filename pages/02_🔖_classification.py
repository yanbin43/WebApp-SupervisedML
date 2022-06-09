import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.graph_objects as go

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#--------------------------------------------------------------------------------------------------------------#

st.title('Machine Learning: Simple Linear Regression')

st.sidebar.title('Start here:')
data_option = st.sidebar.radio(label = 'Select Dataset Type',
                               options = ['Default', 'User-Defined'])

#--------------------------------------------------------------------------------------------------------------#

if data_option == 'Default':
    
    selected_data = st.sidebar.selectbox(label = 'Select a Default Dataset',
                                         options = ['Diabetes', 'California House Value'])
    
    if selected_data == 'Diabetes':
        
        data = datasets.load_diabetes()
        
        X = pd.DataFrame(data.data, columns = data.feature_names)
        y = pd.DataFrame(data.target, columns = ['target'])
        
    elif selected_data == 'California House Value':
        
        data = datasets.fetch_california_housing()
        
        X = pd.DataFrame(data.data, columns = data.feature_names)
        y = pd.DataFrame(data.target, columns = data.target_names)

    st.header(f'Dataset Chosen: {selected_data}')
    descr = st.expander('About the Dataset')
    descr.write(data.DESCR)


#--------------------------------------------------------------------------------------------------------------#

else:
    
    uploaded_data = st.sidebar.file_uploader(label = 'Upload a csv file',
                                             type = 'csv')
    
    if uploaded_data == None:
        
        st.title('Please upload a dataset to continue.')
        
    else:

        data = pd.read_csv(uploaded_data)
        cols = list(data.columns)
        
        y_name = st.sidebar.selectbox(label = 'Select the target (y)', options = sorted(cols))
        y = data[y_name]
        
        cols.remove(y_name)
        X_names = st.sidebar.multiselect(label = 'Select the features (X)', options = sorted(cols))
        X = data[X_names]
        
        st.header(f'Dataset Chosen: {uploaded_data.name}')
        

#--------------------------------------------------------------------------------------------------------------#

test_size = st.sidebar.number_input(label = 'Set a Test Size', min_value = 0.05, max_value = 0.40, value = 0.20, step = 0.05)

random_state = st.sidebar.number_input(label = 'Set a Random State', min_value = 0, max_value = 10000, value = 100, step = 1)


#--------------------------------------------------------------------------------------------------------------#

try:
    
    st.header('Dataset Overview')
    data_full = pd.concat([X, y], axis = 1)
    st.write(data_full.describe())


    #--------------------------------------------------------------------------------------------------------------#

    st.header('Correlation Heatmap')
    corr_heatmap, ax = plt.subplots()
    ax = sns.heatmap(data_full.corr('pearson'), cmap = 'PuOr', fmt = '.2f', annot = True, vmin = -1, vmax = 1, center = 0)
    st.pyplot(corr_heatmap)

    # corr_heatmap = go.Figure()
    # corr_heatmap.add_trace(go.Heatmap(z = data_full.corr(),
    #                                   x = list(data_full.columns),
    #                                   y = list(data_full.columns),
    #                                   zmax = 1, zmid = 0, zmin = -1,
    #                                   colorscale = 'PuOr'))
    # corr_heatmap.update_xaxes(side = 'top')
    # st.plotly_chart(corr_heatmap)


    #--------------------------------------------------------------------------------------------------------------#

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X_test)

    st.header('Linear Regression Report')

    met1, met2 = st.columns(2)
    met1.metric(label = 'Root Mean Squared Error', value = f'{mean_squared_error(y_test, y_pred)**0.5:.4f}')
    met2.metric(label = 'R-squared', value = f'{r2_score(y_test, y_pred):.4f}')


#--------------------------------------------------------------------------------------------------------------#

except:
    
    st.subheader('Hold on. Target and features are not selected yet.')
