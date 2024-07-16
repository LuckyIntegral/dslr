import pandas as pd


class LogisticRegression:
    '''Logistic Regression model'''
    def __init__(self, df: pd.DataFrame) -> None:
        '''Initialize the model'''
        self.df = df
        self.weights = None

    def train(self) -> None:
        '''Train the model'''
        pass

    def save_model(self) -> None:
        '''Save the model'''
        pass

    def load_model(self) -> None:
        '''Load the model'''
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        '''Make predictions'''
        pass
