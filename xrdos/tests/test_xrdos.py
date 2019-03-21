import unittest

import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing

import xrdos


class test_xrdos(unittest.TestCase):

    def test_split(self):
        data = {'column1': [2, 2, 3], 'column2': [1, 3, 5]}
        df = pd.DataFrame(data)
        one, two = xrdos.split(df, 1)
        assert one[0] == 1
        assert two[0] == 2
        return

    def test_scaling(self):
        data = {'column1': [2.0, 2.0, 3.0], 'column2': [1.0, 3.0, 5.0]}
        df = pd.DataFrame(data)
        df, scaler = xrdos.scaling(df)
        assert df.loc[0].iloc[0] == 0
        assert df.loc[2].iloc[0] == 1
        return

    def test_linear_regression(self):
        regr = xrdos.linear_regression()
        x = np.array([0.5, 1.0, 2.0])
        y = np.array([0.5, 1.0, 2.0])
        regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        p = np.array([0.5, 1.0, 2.0]).reshape(-1, 1)
        prediction = regr.predict(p)
        for i in range(len(prediction)):
            assert int(prediction[i]) == int(x[i])
        assert isinstance(p, np.ndarray)
        return

    def test_neural_network(self):
        assert isinstance(
            xrdos.neural_network(1),
            keras.wrappers.scikit_learn.KerasRegressor)
        return

    def test_split_and_scale(self):
        data = {'column1': [2, 2, 3], 'column2': [1, 3, 5]}
        df = pd.DataFrame(data)
        x, y, z = xrdos.split_and_scale(df, 1, (False, 1, False))
        assert x[0] == 1
        assert y.iloc[2].iloc[0] == 1
        return

    def test_polynomialize(self):
        data = {'column1': [2, 2, 3], 'column2': [1, 3, 5]}
        df = pd.DataFrame(data)
        yes = [True, 2, True]
        poly = xrdos.polynomialize(df, yes)
        print(type(poly))
        assert poly[0, 0] == 1
        assert poly[2, 3] == 15
        assert isinstance(poly, np.ndarray)
        return

    def test_train_model(self):
        data = {'column1': [2, 2, 3], 'column2': [1, 3, 5]}
        df = pd.DataFrame(data)
        df1 = pd.DataFrame(data)
        model, accuracy, scaler = xrdos.train_model(
            df, df1, xrdos.linear_regression(), 1, [False, 1, False])
        a = np.array(df.iloc[0]).reshape(-1, 1)
        assert int(model.predict(scaler.transform(a))[0][0]) == 2
        assert isinstance(model, sklearn.linear_model.base.LinearRegression)
        return

    def test_model_prediction(self):
        data = {
            'column1': [
                2, 3, 4], 'column2': [
                1, 3, 5], 'column3': [
                1, 5, 10]}
        df = pd.DataFrame(data)
        properties, predictors = xrdos.split(df, 1)
        predictors = pd.DataFrame(predictors)
        model = xrdos.linear_regression()
        fitted_model = model.fit(predictors, properties)
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(properties)
        prediction, accuracy = xrdos.model_prediction(
            df, fitted_model, scaler, 1, [False, 1, True])
        assert int(prediction[0][0]) == -2
        return
