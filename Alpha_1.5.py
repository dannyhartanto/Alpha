import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import seaborn as sns
import time
from math import sqrt

st.title("Alpha: 1.5")
st.markdown('An easy to use Python web application to develop prediction model (regression) based on exisitng historical data.')
st.markdown('**Developer contact:** dannyhartantodjarum@gmail.com')

st.sidebar.title('Settings')
st.sidebar.markdown('Change and manipulate your prediction parameters here')

#Upload a file
st.set_option('deprecation.showfileUploaderEncoding', False)
st.subheader('Dataset:')
filename = st.file_uploader("Upload a file", type=("csv"))

#Result to display
st.sidebar.subheader('Display:')
full_dataset = st.sidebar.checkbox('Full Data Set', value=False, key=None)
missing_data = st.sidebar.checkbox('Missing Data', value=False, key=None)
data_distribution = st.sidebar.checkbox('Data Distribution', value=False, key=None)
custom_graph = st.sidebar.checkbox('Custom Graph [BETA]', value=False, key=None)
corr_analysis = st.sidebar.checkbox('Correlation Analysis', value=False, key=None)
pp_graph = st.sidebar.checkbox('Pair Plot Graph', value=False, key=None)
hm_graph = st.sidebar.checkbox('Heat Map Correlation', value=False, key=None)
check_graph = st.sidebar.checkbox('Actual vs Predicted Graph', value=False, key=None)

#Subheader model settings
st.sidebar.subheader('Model Settings')

#User remove non-numeric columns
remove_string = st.sidebar.checkbox('Remove non-numeric paramerter', value=True, key=None)
#Label encoding
encode_string = st.sidebar.checkbox('Label encoding [BETA]', value=False, key=None)
#User set zero to NaN
zero_nan = st.sidebar.checkbox('Replace zero with median', value=False, key=None)
#Develop prediction model
train_model = st.sidebar.checkbox('Develop prediction model', value=False, key=None)
#Grid search - Auto tuning
auto_tuning = st.sidebar.checkbox('Hyperparameter Tuning', value=False, key=None)
#Custom Prediction
custom_predict = st.sidebar.checkbox('Custom Prediction', value=False, key=None)

#Test data set size - User Input
test_data = st.sidebar.number_input('Test Data Size (%)',0,100,20)
test_data = test_data * 0.01

#Data Scaler Selection
scaler_name = st.sidebar.selectbox('Select Data Scaler',('Standard Scaler', 'Normalizer', 'Min-Max Scaler', 'Quantile Transformer', 'Power Transformer', 'Robust Scaler', 'None'))

#Data reduction selection
reduction_name = st.sidebar.selectbox('Select Data Reduction Technique',('None', 'PCA', 'LDA', 'ICA', 'SVD'))

#Number of dimension - data reduction
no_dimen = st.sidebar.number_input("No of Dimensions:",min_value=3)

#Prediction Model Selection
regressor_name = st.sidebar.selectbox('Select Prediction Model',('Multi linear Regression', 'Random Forest Regression', 'Extra Tree Regression', 'Decision Tree + AdaBoost'))

#Define model parameter
def get_model_param(rgsr_name):
    params = dict()
    if rgsr_name == "Multi linear Regression":
        n_jobs = st.sidebar.slider('n_jobs',1,10,1)
        params["n_jobs"] = n_jobs
    elif rgsr_name == "Random Forest Regression":
        n_estimators = st.sidebar.slider('n_estimators',10,1000,100)
        min_samples_split = st.sidebar.slider('min_samples_split', 2, 50, 2)
        params["n_estimators"] = n_estimators
        params["min_samples_split"] = min_samples_split
    elif rgsr_name == "Extra Tree Regression":
        n_estimators = st.sidebar.slider('n_estimators', 10, 1000, 100)
        min_samples_split = st.sidebar.slider('min_samples_split', 2, 50, 2)
        params["n_estimators"] = n_estimators
        params["min_samples_split"] = min_samples_split
    else:
        n_estimators = st.sidebar.slider('n_estimators', 10, 1000, 50)
        params["n_estimators"] = n_estimators
    return params

params = get_model_param(regressor_name)

#Upload File
if filename == None:
    st.warning('No file selected.')
    st.sidebar.warning("Data not detected. Please upload a csv file first!")
    st.stop()
else:
    st.success('File successfully loaded')

#Name the selected file
csv_data = pd.read_csv(filename)

#Remove non-numeric columns
if remove_string == True:
    csv_data = csv_data.select_dtypes(include=None, exclude=object)

#Convert data type to float
##csv_data = csv_data.astype(float)

#Subheader
st.subheader('Parameters Selection:')
st.markdown('Parameters from your csv file are automatically detected. Select your parameters here.')

#User select output param
sorted_param_out = sorted(csv_data)
out_param = st.selectbox('Output Parameters',sorted_param_out)

#User select input parameters
param_input = csv_data.drop(out_param, axis=1)
sorted_param_in = sorted(param_input)
in_param = st.multiselect('Input Parameters',sorted_param_in, sorted_param_in)

if csv_data[in_param].empty:
    st.warning('Please select input parameters')
    st.stop()

#Print number of columns
##st.subheader('No of input parameters:')
no_col = len(in_param)
##st.write(no_col)

#Column names
input_name = list(in_param)
##st.write(in_param)

#User input
if custom_predict == True:
    st.sidebar.subheader('Custom Prediction')
    st.sidebar.markdown('Enter input parameters below for custom prediction')
    st.subheader('Custom Input Parameters')
    if remove_string == True:
        user_data = []
        for n in range(no_col):
            user_input = st.sidebar.number_input(input_name[n])
            user_data.append(user_input)

        user_data = pd.DataFrame(user_data)
        user_data = user_data.T
        #Naming the user_data column
        user_data.columns = [in_param]
        st.write(user_data)
    else:
        st.warning('Custom prediction currently does not work with label encoding. Please remove non-numeric parameters')

#bottom sidebar UI space
st.sidebar.subheader('')

#Megre input and output parameters
csv_data = pd.concat([csv_data[in_param], csv_data[out_param]],axis='columns')

#Shape of dataset
st.subheader('Data Shape:')
st.write(csv_data.shape)

#If user select set zero to median
if zero_nan == True:
    #List zero values sum
    st.subheader('Sum of zero values:')
    zero_sum = (csv_data == 0).sum()
    st.write(zero_sum)

    #User input set zero to median
    param_zero = csv_data
    sorted_param_zero = sorted(param_zero)
    zero_param = st.multiselect('Parameter with zero values to be replaced with median', sorted_param_zero)

    #Convert zero to NaN
    csv_data[zero_param] = csv_data[zero_param].replace(0, np.NaN)

#Display Data Head
if full_dataset == False:
    st.subheader('Data Head (100 rows):')
    st.write(csv_data.head(n=100))

#Display full data set
if full_dataset == True:
    st.subheader('Full Data Set:')
    st.dataframe(csv_data)

#Label encoding
if remove_string == False and encode_string == True:
    st.subheader('Label Encoding')
    le = preprocessing.LabelEncoder()
    encode_param = csv_data.select_dtypes(include=object, exclude=None)
    sorted_encode_param = sorted(encode_param)
    to_numeric = st.multiselect('Parameter to be converted to numeric values (do not select string type variable here!):', sorted_encode_param)
    csv_data[to_numeric] = csv_data[to_numeric].apply(pd.to_numeric, errors='coerce')
    encode_param = csv_data.select_dtypes(include=object, exclude=None)
    sorted_encode_param = sorted(encode_param)
    st.write(encode_param.dtypes)
    try:
        for i in range(0, len(sorted_encode_param)):
            csv_data[sorted_encode_param[i]] = le.fit_transform(csv_data[sorted_encode_param[i]])
        st.success('All object type variables have been successfully encoded')
    except TypeError:
        st.warning("Some of the object variables are of mixed type (strings + numbers). Please convert them to numeric values in order to proceed.")
        st.stop()
elif remove_string == True and encode_string == True:
    st.subheader('Label Encoding')
    st.warning("Choose only either remove non-numeric parameter or label encoding")

#Display percentange missing data
if missing_data == True:
    st.subheader('Missing Data:')
    st.markdown('Missing data will be automatically imputed with the median values of the parameter.')
    st.write((csv_data.isna()).sum())

#Show statistics on the data
st.subheader('Data Statistics:')
st.write(csv_data.describe())

#Remove outlier
#csv_data = csv_data[csv_data.CO <= 4]
#csv_data = csv_data[csv_data.PM2_5 <=120]
#csv_data = csv_data[csv_data.PM10 <=120]

#Data distribution
if data_distribution == True:
    st.subheader('Data Distribution:')
    csv_data.hist(bins=50, color='#339E98', figsize=[20,15])
    plt.show()
    with st.spinner('Plotting...'):
        st.pyplot()

#Custom Graph
if custom_graph == True:
    st.subheader('Custom Graph [BETA]:')
    custom_plot = csv_data
    custom_plot = sorted(custom_plot)
    graph_type = st.selectbox('Select graph type:',('Scatter', 'Line (mean)', 'Bar (sum)', 'Box Plot'))
    custom_x = st.selectbox('Select X variable:', custom_plot)
    custom_y = st.selectbox('Select y variable:', custom_plot)
    figure_size = st.slider('Figure size:', 5,50,10)
    if csv_data[custom_x].empty and csv_data[custom_y].empty:
        st.warning('Please select parameters to plot')
    else:
        with st.spinner('Plotting...'):
            plt.figure(figsize=[figure_size*2, figure_size])
            if graph_type == "Line (mean)":
                y_axis = csv_data[custom_y].groupby(csv_data[custom_x]).mean()
                plt.plot(y_axis.index.values, y_axis)
                st.write(y_axis)
            elif graph_type == "Bar (sum)":
                y_axis = csv_data[custom_y].groupby(csv_data[custom_x]).sum()
                plt.bar(y_axis.index.values, y_axis)
                st.write(y_axis)
            elif graph_type == "Box Plot":
                sns.boxplot(y= custom_y, x= custom_x, data= csv_data, width=0.5, palette="colorblind")
            else:
                plt.scatter(csv_data[custom_x], csv_data[custom_y], alpha=0.1)
            plt.xlabel(custom_x)
            plt.ylabel(custom_y)
            plt.title(custom_y + " v.s " + custom_x)
            plt.plot()
            st.pyplot()

#Correlation Analysis
if corr_analysis == True:
    st.subheader('Correlation Analysis:')

    if remove_string == True or encode_string == True:
        corr_matrix = csv_data.corr()
        st.write(corr_matrix[out_param].sort_values(ascending=False))
    else:
        st.warning('Please remove non-numeric parameters or perform label encoding')

#HeatMap
if hm_graph == True:
    corr = csv_data.corr()
    plt.figure(figsize=(20, 12))
    sns.heatmap(corr, annot=True, cmap="BuGn")
    st.subheader('Heat Map Correlation:')
    with st.spinner('Plotting...'):
        st.pyplot()

#Simple imputer to patch in missing data with median value and converting data type to float
st.subheader('Imputing Missing Values:')
if remove_string == True or encode_string == True:
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    imputer.fit(csv_data)
    X = imputer.transform(csv_data)
    csv_data = pd.DataFrame(X, columns=csv_data.columns)

    # RESET PANDA INDEX AFTER DATA CLEANING
    csv_data = csv_data.reset_index(drop=True)
    st.success('Missing values successfully imputed')
else:
    st.warning('Please remove non-numeric parameters or perform label encoding')


#Display pairplot graph
if pp_graph == True:
    st.subheader('Pairplot Graph')
    if remove_string == True or encode_string == True:
        pair_plot = csv_data
        pair_plot = sorted(pair_plot)
        pp_select = st.multiselect('Select parameters to be plotted:', pair_plot)
        if csv_data[pp_select].empty:
            st.warning('Please select parameters to plot')
        else:
            sns.pairplot(csv_data[pp_select], diag_kind="kde")
            with st.spinner('Plotting...'):
                st.pyplot()
    else:
        st.warning('Please remove non-numeric parameters or perform label encoding')


#Split the data into X and Y
if train_model == True:
    st.header('Prediction Model:')
    if remove_string == True or encode_string == True:
        y = csv_data[out_param]
        X = csv_data[in_param]
    else:
        st.warning('Please remove non-numeric parameters or perform label encoding')
        st.stop()
else:
    st.stop()


#Split the data into test and training set
from sklearn.model_selection import train_test_split
train_prepared, X_test, train_labels, y_test = train_test_split(X, y, test_size=test_data, random_state=1234)

#Define the selected data scaler
def get_scaler(scl_name):
    if scl_name == "Standard Scaler":
        scaler = preprocessing.StandardScaler()
    elif scl_name == "Normalizer":
        scaler = preprocessing.Normalizer()
    elif scl_name == "Min-Max Scaler":
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    elif scl_name == "Quantile Transformer":
        scaler = preprocessing.QuantileTransformer()
    elif scl_name == "Power Transformer":
        scaler = preprocessing.PowerTransformer()
    elif scl_name == "Robust Scaler":
        scaler = preprocessing.RobustScaler()
    else:
        scaler = None
    return scaler

scaler = get_scaler(scaler_name)

# Fit your data on the scaler object
try:
    train_prepared = scaler.fit_transform(train_prepared)
    X_test = scaler.transform(X_test)
except AttributeError:
    train_prepared = train_prepared
    X_test = X_test
except ValueError:
    st.warning(scaler_name + " does not work this data set. Please select a different data scaler.")
    st.warning('If issues persisted, please contact developer')
    st.stop()

#Data reduction Technique
def get_reduction(data_reduc):
    if data_reduc == "PCA":
        reduction = PCA(n_components=no_dimen)
    elif data_reduc == "LDA":
        reduction = LinearDiscriminantAnalysis(n_components=no_dimen)
    elif data_reduc == "ICA":
        reduction = FastICA(n_components=no_dimen)
    elif data_reduc == "SVD":
        reduction = TruncatedSVD(n_components=no_dimen, n_iter=10, random_state=42)
    else:
        reduction = None
    return reduction

reduction = get_reduction(reduction_name)

# Perform data reduction
if reduction_name == "LDA":
    train_labels = train_labels.astype('int')
    train_prepared = reduction.fit_transform(train_prepared, train_labels)
    X_test = reduction.transform(X_test)
elif reduction_name == "None":
    train_prepared = train_prepared
    X_test = X_test
else:
    train_prepared = reduction.fit_transform(train_prepared)
    X_test = reduction.transform(X_test)


#Define the selected prediction model
def get_regresson(rgsr_name, params):
    if rgsr_name == "Multi linear Regression":
        model = LinearRegression(n_jobs=params["n_jobs"])
    elif rgsr_name == "Random Forest Regression":
        model = RandomForestRegressor(n_estimators=params["n_estimators"], min_samples_split=params["min_samples_split"])
    elif rgsr_name == "Extra Tree Regression":
        model = ExtraTreesRegressor(n_estimators=params["n_estimators"], min_samples_split=params["min_samples_split"], random_state=1234)
    else:
        model = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=params["n_estimators"], learning_rate=1)
    return model

model = get_regresson(regressor_name, params)

#Create and train the model
with st.spinner('Calculating...'):
    time_start = time.time()
    model.fit(train_prepared, train_labels)
    time_end = time.time()
    model_score = model.score(X_test, y_test)
    model_score = model_score * 100
    model_score = str(round(model_score, 2))

#Display the R2 values
st.subheader('Test Accuracy Score:')
st.write(model_score + '%')


#Display RMSE
st.subheader('RMSE:')
rmse_text = sqrt(mean_squared_error(y_test, model.predict(X_test)))
rmse_text = str(round(rmse_text, 2))
st.write(rmse_text)


#Display model training time
training_time = time_end - time_start
training_time = str(round(training_time,2))
st.subheader('Training Time')
st.write(training_time + ' s')

#Display actual vs predicted graph
if check_graph == True:
    model_pred = model.predict(X_test)
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.scatter(y_test, model_pred, color='#339E98')
    m, b = np.polyfit(y_test, model_pred, 1)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title(regressor_name)
    plt.plot(y_test, m * y_test + b, color='#F63366')
    st.subheader('Actual vs Predicted Graph:')
    with st.spinner('Plotting...'):
        st.pyplot()


#Grid search (Parameter Tuning)
if auto_tuning == True:
    st.subheader('Parameter Tuning:')
    if st.button('Run Grid Search'):
        if regressor_name == 'Random Forest Regression' or regressor_name == 'Extra Tree Regression':
            param_grid = [{'n_estimators': [60, 80, 100, 120], 'min_samples_split': [2, 5, 8, 10]}]
        elif regressor_name == 'Multi linear Regression':
            param_grid = [{'n_jobs': [1, 2, 4, 8, 10]}]
        else:
            param_grid = [{'n_estimators': [60, 80, 100, 120]}]
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(train_prepared, train_labels)
        st.write(grid_search.best_params_)


#Custom prediction
if custom_predict == True:
    if st.sidebar.button('Predict output'):
        st.subheader('Predicted Output:')
        #Scale the data
        if scaler_name == 'None':
            user_data = user_data
        else:
            user_data = scaler.transform(user_data)
        #Reduce the data
        if reduction_name == 'None':
            user_data = user_data
        else:
            user_data = reduction.transform(user_data)
        # Predict output
        prediction = model.predict(user_data)
        prediction = str(round(prediction[0], 2))
        st.write(out_param + ' = ' + prediction)

st.sidebar.subheader('')