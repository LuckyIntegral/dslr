from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser, Namespace
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

HOUSES = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']
DEFAULT_OUTPUT_PATH = 'houses.csv'


def parse_arguments() -> Namespace:
    """ Parse command-line arguments """
    parser = ArgumentParser(usage="predict.py -d <dataset> -m <model>")
    parser.add_argument('-d', type=str, required=True, help='Path to processed dataset CSV')
    parser.add_argument('-m', type=str, required=True, help='Path to trained model weights (pickle)')
    parser.add_argument('-o', type=str, default=DEFAULT_OUTPUT_PATH, help='Output file path for predictions')

    return parser.parse_args()


def load_model(path: str) -> pd.DataFrame:
    """ Load trained weights from a pickle file """
    return pd.read_pickle(path)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """ Compute the sigmoid function """
    return 1 / (1 + np.exp(-z))


def hypothesis(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    """ Calculate the hypothesis using the sigmoid function """
    return sigmoid(np.dot(features, weights.T))


def predict(features: np.ndarray, weights: pd.DataFrame) -> pd.DataFrame:
    """ Generate predictions for each sample in the dataset """
    predictions = pd.DataFrame(columns=HOUSES)
    res = pd.DataFrame(columns=["Hogwarts House"])

    for house in HOUSES:
        predictions[house] = hypothesis(weights.loc[house].values, features)

    res['Hogwarts House'] = predictions.idxmax(axis=1)
    res.index.name = 'Index'

    return res


def load_and_preprocess_data(csv_path: str) -> np.ndarray:
    """ Load the dataset from CSV and preprocess features """
    df = pd.read_csv(csv_path)
    features_df = df.drop('Hogwarts House', axis=1)

    scaler = StandardScaler()
    return scaler.fit_transform(features_df)


def main() -> None:
    """ Entrypoint for generating and saving predictions """
    args = parse_arguments()

    logging.info(f"Loading model weights from: {args.m}")
    weights = load_model(args.m)

    logging.info(f"Loading dataset from: {args.d}")
    features = load_and_preprocess_data(args.d)

    logging.info(f"Predicting using dataset: {args.d}")
    predictions = predict(features, weights)

    logging.info(f"Saving predictions to: {args.o}")
    predictions.to_csv(args.o)
    logging.info("Prediction completed successfully.")


if __name__ == '__main__':
    main()
