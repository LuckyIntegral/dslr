import pandas as pd
import numpy as np
import sklearn.ensemble
from utils import HOUSES, FEATURES_TO_DROP, MODEL_SETUPS
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
from describe import ft_mean, ft_std


class LogisticRegression:
    '''Logistic Regression model'''
    def __init__(self, path_to_data: str) -> None:
        '''Initialize the model'''
        df = pd.read_csv(path_to_data)
        self.X, self.Y = self.parse_csv(df)
        self.weights = pd.DataFrame(columns=self.X.columns)
        self.LR = MODEL_SETUPS['learning_rate']
        self.epochs = MODEL_SETUPS['epochs']

    def hypothesis(self, weights: np.array, features: np.array) -> np.array:
        '''Hypothesis function
        args:
            weights: weights of the model
            features: features
        return:
            predicted values
        '''
        return 1 / (1 + np.exp(-np.dot(features, weights.T)))

    def one_vs_all(self, house: str) -> np.array:
        '''Train the model for one vs all
        args:
            house: house to train the model
        return:
            weights: weights of the model
        '''
        weights = np.ones(shape=(self.X.shape[1]))
        actual = np.where(self.Y == house, 1, 0)
        for _ in range(self.epochs):
            predicted = self.hypothesis(weights, self.X)
            error = predicted - actual
            gradient = np.dot(error, self.X) / self.X.shape[0]
            weights = weights - self.LR * gradient
        return weights

    def train(self) -> None:
        '''Train the model'''

        for house in HOUSES:
            weights = self.one_vs_all(house)
            self.weights.loc[house] = weights

        print(f'Accuracy: {self.accuracy()}')

    def accuracy(self) -> float:
        '''Compute the accuracy of the model
        return:
            accuracy: accuracy of the model
        '''
        predictions = pd.DataFrame(columns=HOUSES)
        for house in HOUSES:
            weights = self.weights.loc[house]
            predictions[house] = self.hypothesis(weights, self.X)
        predictions['Hogwarts House'] = predictions.idxmax(axis=1)
        self.Y.reset_index(drop=True, inplace=True)
        return (predictions['Hogwarts House'] == self.Y).mean() * 100

    def save_model(self, filename: str) -> None:
        '''Store the model'''
        pd.to_pickle(self.weights, filename)

    def load_model(self, filename: str) -> None:
        '''Load the model'''
        self.weights = pd.read_pickle(filename)

    def predict(self, dataset: str) -> pd.DataFrame:
        '''Predict the house of the students
        args:
            dataset: path to the dataset
        return:
            predictions: predicted house of the students
        '''
        df = pd.read_csv(dataset)
        X = df.drop(FEATURES_TO_DROP, axis=1).drop('Hogwarts House', axis=1)
        X = X.apply(lambda col: self.standardize(col))
        predictions = pd.DataFrame(columns=HOUSES)
        for house in HOUSES:
            weights = self.weights.loc[house]
            predictions[house] = self.hypothesis(weights, X)
        # predictions['Hogwarts House'] = predictions.idxmax(axis=1)
        return predictions.idxmax(axis=1)


    def parse_csv(self, df: pd.DataFrame) -> tuple:
        '''Parse the dataframe to get features and target
        args:
            df: dataframe to parse
        return:
            X: features
            Y: target
        '''
        df.drop(FEATURES_TO_DROP, axis=1, inplace=True)
        df.dropna(inplace=True)

        X = df.drop('Hogwarts House', axis=1)
        X = X.apply(lambda col: self.standardize(col))
        Y = df['Hogwarts House']
        return X, Y

    def standardize(self, col: np.array) -> np.array:
        '''standardize the column
        args:
            col: column to standardize
        return:
            standardized column with mean 0 and std 1
        '''
        mean = ft_mean(col)
        std = ft_std(col)
        return (col - mean) / std
