import pandas as pd
import numpy as np
import sklearn.ensemble
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn

HOUSES = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']

FEATURES_TO_DROP = ['Care of Magical Creatures', 'Arithmancy', 'Defense Against the Dark Arts', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index']

MODEL_SETUPS = {
    'learning_rate': 0.1,
    'epochs': 10000,
}

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
        return 1 / (1 + np.exp(-np.dot(features, weights.T)))

    def one_vs_all(self, house: str) -> np.array:
        weights = np.ones(shape=(self.X.shape[1]))
        actual = np.where(self.Y == house, 1, 0)
        for _ in range(self.epochs):
            predicted = self.hypothesis(weights, self.X)
            error = predicted - actual
            gradient = np.dot(error, self.X) / self.X.shape[0]
            weights = weights - self.LR * gradient
        return weights

    def train(self) -> None:

        for house in HOUSES:
            weights = self.one_vs_all(house)
            self.weights.loc[house] = weights

        print(f'Accuracy: {self.accuracy()}')

    def accuracy(self) -> float:
        predictions = pd.DataFrame(columns=HOUSES)
        for house in HOUSES:
            weights = self.weights.loc[house]
            predictions[house] = self.hypothesis(weights, self.X)
        predictions['Hogwarts House'] = predictions.idxmax(axis=1)
        self.Y.reset_index(drop=True, inplace=True)
        return (predictions['Hogwarts House'] == self.Y).mean() * 100

    def save_model(self, filename: str) -> None:
        pd.to_pickle(self.weights, filename)

    def load_model(self, filename: str) -> None:
        self.weights = pd.read_pickle(filename)

    def predict(self, dataset: str) -> pd.DataFrame:
        res = pd.DataFrame(columns=["Hogwarts House"])
        df = pd.read_csv(dataset)
        df.fillna(0, inplace=True)
        X = df.drop(FEATURES_TO_DROP, axis=1).drop('Hogwarts House', axis=1)
        X = X.apply(lambda col: self.standardize(col))
        predictions = pd.DataFrame(columns=HOUSES)
        for house in HOUSES:
            weights = self.weights.loc[house]
            predictions[house] = self.hypothesis(weights, X)

        res["Index"] = df["Index"]
        res['Hogwarts House'] = predictions.idxmax(axis=1)
        res.set_index('Index', inplace=True)
        return res


    def parse_csv(self, df: pd.DataFrame) -> tuple:
        df.drop(FEATURES_TO_DROP, axis=1, inplace=True)
        # df.dropna(inplace=True)
        df.fillna(0, inplace=True) # fill NaN values with 0, no way it works better

        X = df.drop('Hogwarts House', axis=1)
        X = X.apply(lambda col: self.standardize(col))
        Y = df['Hogwarts House']
        return X, Y

    def standardize(self, col: np.array) -> np.array:
        mean = np.mean(col)
        std = np.std(col)
        return (col - mean) / std
