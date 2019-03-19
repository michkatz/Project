# code goes here
# everything in this file will be available to the xrdos namespace
# automatically


import pandas as pd
import numpy as np
import sklearn
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing, metrics 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Input, Dense, Activation
from keras import optimizers
from keras import regularizers
from keras.regularizers import Regularizer


#NOTE: none of these fxns are meant for single point input, input should always be a pd.DataFrame with more
#than one row, not an unreasonable request since sklearn makes you reshape 1D arrays.

def single_plot(x, y, figsize, title, xtitle, ytitle, xticks, yticks):
    """Plots a single figure.
    x: x axis data
    y: y axis data
    figsize: tuple of format (xdim, ydim)
    title: tuple of format (title(string), fontsize(int))
    xtitle and ytitle similar to title
    xticks and yticks: tuple of format (array of tick values, fontsize)"""
    fig, ax1 = plt.subplots(figsize = figsize)
    plt.title(title[0], size = title[1])
    plt.xticks(xticks[0], size = xticks[1])
    plt.yticks(yticks[0], size = yticks[1])
    plt.xlabel(xtitle[0], size = xtitle[1])
    plt.ylabel(ytitle[0], size = ytitle[1])
    ax1.set_xlim(0, 6)
    ax1.scatter(x, y, marker = '.', alpha = .7)
    return


def split_and_scale(df, n, yes):
    """Splits training dataframe into predictors and properties to be predicted and returns them in 2 new dfs.
       This function assumes all of the predictors are grouped together on the right side of the df.
       df_train: training df
       n: number of properties to be predicted(number of outputs)"""
    # Splitting into properties and predictors
    properties, predictors = split(df, n)
    # Adding polynomial term columns
    predictors_polynomial = polynomialize(predictors, yes)
    # Scaling predictor data
    predictors_scaled_polynomial, predictors_scaler_polynomial = scaling(predictors_polynomial)
    return properties, predictors_scaled_polynomial, predictors_scaler_polynomial 


def polynomialize(series, yes):
    """Adds polynomial features to degree 3, including interaction features. 
    series: an input ndarray of floats to be polynomialized.
    This function returns a ndarray of all of the features specified above.
    
    series: dataframe to be polynomialized
    yes: list, array or tuple of the form:
    (Bool deciding whether to add polynomial terms, degree of highest polynomial, bool deciding whether to only provide interaction terms)
    Returns the polynomialized series."""
    # Creating polynomial object
    if yes[0]:
        poly = PolynomialFeatures(degree = yes[1], interaction_only = yes[2])
        # Adding polynomial terms
        series = poly.fit_transform(series)
    return series


def split(df, n):
    """Takes an input pd.DataFrame and returns 2 ndarrays of the properties 
    and predictors."""
    properties = df[df.columns[-n:]].values
    predictors = df[df.columns[:-n]].values
    return properties, predictors


def scaling(df_train):
    """This function takes a pd.DataFrame, creates a sklearn.StandardScaler, scales the DataFrame,
       and returns the scaled data in a pd.DataFrame as well as the sklearn.StandardScaler object
       for transforming data back to unscaled form post machine learning.
       df_train: pd.DataFrame(for our purposes should be of shape 20 columns by an arbitrary number of rows)
       
       Returns scaled dataframe and its respective scaler"""
    #Creating scaler object
    scaler = preprocessing.MinMaxScaler()
    #Scaling df_train
    scaled_data = pd.DataFrame(scaler.fit_transform(df_train))
    return scaled_data, scaler


def train_model(df_train, df_validation, model, n, yes):
    """This function takes a training DataFrame, validation DataFrame and a preconfigured model
       and trains said model on the training data followed by measuring error on validation data and 
       returning both the trained model and accuracy metric. This function assumes whatever parameter(s)
       being predicted is in the last column(s) of df_train.
       n: number of outputs
       df_validation: to measure accuracy
       model: pre initialized model object
       yes: list, array or tuple of the form:
       (Bool deciding whether to add polynomial terms, degree of highest polynomial, bool deciding whether to only provide interaction terms)
       because this function returns the trained model, more metrics can be performed later that are specific
       to whatever package it is in/the type of model it is
       Returns the model object, RMSE on validation set and the scaler for predictors
       Note: can only predict data which has been scaled with the scaler this function returns"""
    #generating scaled data and their respective scaler objects
    t_properties, t_predictors_scaled, t_predictors_scaler = split_and_scale(df_train, n, yes)
    v_properties, v_predictors_scaled, v_predictors_scaler = split_and_scale(df_validation, n, yes)
    #supervised learning of predictors and properties to fit model, note: keras does not take pd.DataFrames for
    #training, using .values fixes this
    model.fit(t_predictors_scaled, t_properties)
    #predicting output of validation set
    predictions = pd.DataFrame(model.predict(v_predictors_scaled))
    #calculating RMSE from sklearn package
    val_error = np.sqrt(metrics.mean_squared_error(predictions, v_properties))
    return model, val_error, t_predictors_scaler


def model_prediction(test_data, fitted_model, scaler, n, yes):
    """Takes a fitted model and predicts the output of test data, returns the predicted data and accuracy.
       THIS FUNCTION IS ONLY TO BE USED FOR FUTURE PREDICTIONS OR TESTING(WHICH SHOULD ONLY BE DONE ONCE).
       Do not use this while training a model, that's what the validation data will be used for. We do not 
       want to introduce bias into our model by fitting to the test data
       n = number of predictors"""
    #splitting predictors and properties
    properties, predictors = split(test_data, n)
    predictors = polynomialize(predictors, yes)
    predictors_scaled = scaler.transform(predictors)
    #predicting based on scaled input predictors
    prediction = fitted_model.predict(predictors_scaled)
    #calculating MSE
    accuracy_metric = np.sqrt(metrics.mean_squared_error(properties, prediction))

    return prediction, accuracy_metric


def neural_network(input_dimension):
    """Creates a neural network object to be passed into train_model function, can change properties of net
       here. Alternatively, a neural network object can be created using Keras and passed to the train_model 
       function to be fitted.
       
       input_dimension: the dimensionality of the input predictors"""
    def model():
        model = Sequential()
        model.add(Dense(1, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
        model.add(Dense(20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer = 'normal'))#kernel_initializer = initial values of outputs i think
        opt = optimizers.Nadam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model
    # Creating neural network object
    network = KerasRegressor(build_fn=model, epochs=150, batch_size=1, verbose=1)
    return network


def linear_regression():
    """Creates a linear regression object
       to be passed to the train_model fxn."""
    regr = LinearRegression()
    return regr

def coefficient_statistics(df):
    """Creates a linear regression model using statsmodels, used to get
       p values, confidence intervals and other metadata for models.
       This function has too specific of a use case for a test function."""
    fit_object = smf.ols(formula='band_gap ~ amplitude_0 + amplitude_1 + amplitude_2 + amplitude_3 + amplitude_4 + amplitude_5 + amplitude_6 + amplitude_7 + amplitude_8 + amplitude_9 + two_theta_1+ two_theta_2 + two_theta_3 + two_theta_4 + two_theta_5 + two_theta_6 + two_theta_7 + two_theta_8 + two_theta_9', data=df)
    ft = fit_object.fit()
    return ft.summary()
