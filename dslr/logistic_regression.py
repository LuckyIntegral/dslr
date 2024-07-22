import pandas as pd
import numpy as np
import sklearn.ensemble
from utils import COURSES, HOUSES
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
from describe import ft_mean, ft_std


class LogisticRegression:
    '''Logistic Regression model'''
    def __init__(self, path_to_data: str) -> None:
        '''Initialize the model'''
        df = pd.read_csv(path_to_data)
        self.Y = df['Hogwarts House']
        self.X = self.standardize(df)
        self.result = pd.DataFrame(columns=self.X.columns)
        self.LR = 0.1

    def normalize(self, col: np.array) -> np.array:
        mean = ft_mean(col) # switch to ft_mean
        std = ft_std(col) # switch to ft_std
        return (col - mean) / std

    def h(self, weights: np.array, Xi: np.array) -> np.array:
        print(-np.dot(Xi, weights.T))
        return 1 / (1 + np.exp(-np.dot(Xi, weights.T)))

    def one_vs_all(self, house: str) -> np.array:
        weights = np.ones(shape=(self.X.shape[1]))
        Yi = np.where(self.Y == house, 1, 0)
        for _ in range(1):
            print(weights)
            predicted = self.h(weights, self.X)
            # print(predicted)
        return np.ones(shape=(self.X.shape[1]))

    def train(self) -> None:
        '''Train the model'''

        for house in HOUSES:
            weights = self.one_vs_all(house)
            self.result.loc[house] = weights

    def save_model(self) -> None:
        '''Save the model'''
        pass

    def load_model(self) -> None:
        '''Load the model'''
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        '''Make predictions'''
        pass

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Standardize the data'''
        X = df[COURSES].drop(['Care of Magical Creatures', 'Arithmancy', "Defense Against the Dark Arts"], axis=1)
        X = X.apply(lambda col: self.normalize(col)) # test before debug
        X = X.dropna()
        return X
