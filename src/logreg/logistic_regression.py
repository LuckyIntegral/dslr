import pandas as pd
import numpy as np
import math
from tqdm import tqdm


HOUSES = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']

FEATURES_TO_DROP = ['Care of Magical Creatures', 'Arithmancy', 'Defense Against the Dark Arts', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Index']

MODEL_SETUPS = {
    'learning_rate': 0.1,
}

class LogisticRegression:
    '''Logistic Regression model'''
    def __init__(self, path_to_data: str, is_train: bool = True) -> None:
        '''Initialize the model'''
        df = pd.read_csv(path_to_data)
        if is_train:
            self.X, self.Y = self.parse_csv(df)
        else:
            self.X = self.Y = df
        self.weights = pd.DataFrame(columns=self.X.columns)
        self.LR = MODEL_SETUPS['learning_rate']

    def hypothesis(self, weights: np.array, features: np.array) -> np.array:
        return 1 / (1 + np.exp(-np.dot(features, weights.T)))

    def gradient_descent(self, weights: np.array, features: np.array, actual: np.array) -> np.array:
        for _ in tqdm(range(100_000)):
            predicted = self.hypothesis(weights, features)
            error = predicted - actual
            gradient = np.dot(error, features) / features.shape[0]
            weights = weights - self.LR * gradient

        return weights

    def stochastic_gradient_descent(self, weights: np.array, features: np.array, actual: np.array) -> np.array:
        for _ in tqdm(range(100)):
            for i in range(features.shape[0]):
                xi = features.iloc[i].values
                yi = actual[i]
                predicted = self.hypothesis(weights, xi)
                error = predicted - yi
                gradient = error * xi
                weights = weights - self.LR * gradient

        return weights

    def mini_batch_gradient_descent(self, weights: np.array, features: np.array, actual: np.array) -> np.array:
        batch_size = int(math.sqrt(features.shape[0]))

        for _ in tqdm(range(1_000)):
            for i in range(0, features.shape[0], batch_size):
                xi = features.iloc[i:i + batch_size].values
                yi = actual[i:i + batch_size]
                predicted = self.hypothesis(weights, xi)
                error = predicted - yi
                gradient = np.dot(error, xi) / xi.shape[0]
                weights = weights - self.LR * gradient

        return weights

    def one_vs_all(self, house: str, algo: bool) -> np.array:
        weights = np.ones(shape=(self.X.shape[1]))
        actual = np.where(self.Y == house, 1, 0)

        match (algo):
            case 'stochastic_gradient_descent':
                weights = self.stochastic_gradient_descent(weights, self.X, actual)
            case 'gradient_descent':
                weights = self.gradient_descent(weights, self.X, actual)
            case 'mini_batch_gradient_descent':
                weights = self.mini_batch_gradient_descent(weights, self.X, actual)
            case _ :
                raise ValueError('Invalid algorithm')

        return weights

    def train(self, algo: str) -> None:

        for house in HOUSES:
            weights = self.one_vs_all(house, algo)
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
        if df['Hogwarts House'].isnull().values.any():
            raise ValueError('Missing values in target column')
        df.fillna(0, inplace=True)  # fill NaN values with 0, no way it works better

        X = df.drop('Hogwarts House', axis=1)
        X = X.apply(lambda col: self.standardize(col))
        Y = df['Hogwarts House']
        return X, Y

    def standardize(self, col: np.array) -> np.array:
        mean = ft_mean(col)
        std = ft_std(col)
        return (col - mean) / std

def filter_series(series: pd.Series) -> pd.Series:
    return pd.Series(filter(lambda x: pd.notna(x), series), dtype=np.float64)

def ft_mean(series: pd.Series) -> np.float64:
    filtered = filter_series(series)

    if len(filtered) == 0:
        return np.nan

    return sum(filtered) / len(filtered)


def ft_std(series: pd.Series) -> np.float64:
    filtered = filter_series(series)
    mean = ft_mean(series)

    if mean is np.nan:
        return np.nan

    if len(filtered) == 1:
        return np.nan

    variance = sum((value - mean) ** 2 for value in filtered) / (len(filtered) - 1)
    return variance ** 0.5
