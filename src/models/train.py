from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser, Namespace
from typing import Callable, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from predict import sigmoid, hypothesis, predict
import pandas as pd
import numpy as np
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

HOUSES = ['Gryffindor', 'Ravenclaw', 'Hufflepuff', 'Slytherin']


@dataclass
class TrainingConfig:
    learning_rate: float = 0.1
    gd_epochs: int = 100_000
    sgd_epochs: int = 100
    mbgd_epochs: int = 1_000


def gradient_descent(weights: np.ndarray,
                     features: np.ndarray,
                     actual: np.ndarray,
                     config: TrainingConfig) -> np.ndarray:
    """ Perform batch gradient descent """
    m = features.shape[0]

    for _ in tqdm(range(config.gd_epochs), desc="Batch GD"):
        predicted = hypothesis(weights, features)
        error = predicted - actual
        gradient = np.dot(error, features) / m
        weights -= config.learning_rate * gradient

    return weights


def stochastic_gradient_descent(weights: np.ndarray,
                                features: np.ndarray,
                                actual: np.ndarray,
                                config: TrainingConfig) -> np.ndarray:
    """ Perform stochastic gradient descent """
    m = features.shape[0]

    for _ in tqdm(range(config.sgd_epochs), desc="SGD Epochs"):
        for i in range(m):
            xi = features[i]
            yi = actual[i]
            predicted = sigmoid(np.dot(xi, weights))
            error = predicted - yi
            gradient = error * xi
            weights -= config.learning_rate * gradient

    return weights


def mini_batch_gradient_descent(weights: np.ndarray,
                                features: np.ndarray,
                                actual: np.ndarray,
                                config: TrainingConfig) -> np.ndarray:
    """ Perform mini-batch gradient descent """
    m = features.shape[0]
    batch_size = int(math.sqrt(m))

    for _ in tqdm(range(config.mbgd_epochs), desc="Mini-batch GD Epochs"):
        for i in range(0, m, batch_size):
            xi = features[i: i + batch_size]
            yi = actual[i: i + batch_size]
            predicted = hypothesis(weights, xi)
            error = predicted - yi
            gradient = np.dot(error, xi) / xi.shape[0]
            weights -= config.learning_rate * gradient

    return weights


def get_optimization_algorithm_function(algo: str):
    """ Return the appropriate gradient descent function based on the algorithm name """
    algorithms = {
        'stochastic_gradient_descent': stochastic_gradient_descent,
        'gradient_descent': gradient_descent,
        'mini_batch_gradient_descent': mini_batch_gradient_descent,
    }

    if algo not in algorithms:
        raise ValueError(f"Invalid algorithm: {algo}")

    return algorithms[algo]


def load_and_preprocess_data(csv_path: str) -> Tuple[np.ndarray, pd.Series]:
    """ Load the dataset from CSV and preprocess features """
    df = pd.read_csv(csv_path)
    labels = df['Hogwarts House']
    features_df = df.drop('Hogwarts House', axis=1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    return features_scaled, labels


def train_one_vs_all(algorithm_function: Callable,
                     house: str,
                     features: np.ndarray,
                     labels: pd.Series,
                     config: TrainingConfig) -> np.array:
    """ Train a one-vs-all classifier for a specific house """
    weights = np.ones(shape=(features.shape[1]))
    actual = np.where(labels == house, 1, 0)

    return algorithm_function(weights, features, actual, config)


def train(algo: str,
          features: np.ndarray,
          labels: pd.Series,
          config: TrainingConfig) -> pd.DataFrame:
    """ Train the model for all houses using the specified gradient descent algorithm """
    algorithm = get_optimization_algorithm_function(algo)
    weights_dict = {}

    for house in HOUSES:
        logging.info(f"Training for house: {house}")
        weights_dict[house] = train_one_vs_all(algorithm, house, features, labels, config)

    return pd.DataFrame(weights_dict, index=[f"feature_{i}" for i in range(features.shape[1])]).T


def parse_arguments() -> Namespace:
    """Parse command line arguments"""
    parser = ArgumentParser(usage='train.py -d <path to your data> -a <algorithm>')
    parser.add_argument('-d', type=str, required=True, help='Path to processed dataset')
    parser.add_argument('-a', type=str, default='stochastic_gradient_descent',
                        choices=['gradient_descent', 'stochastic_gradient_descent', 'mini_batch_gradient_descent'],
                        help='Algorithm to use for training')

    return parser.parse_args()


def compute_accuracy(weights: pd.DataFrame, features: np.ndarray, labels: pd.Series) -> float:
    """ Calculate the accuracy of the model on the provided dataset """
    predictions_df = predict(features, weights)
    matches = predictions_df['Hogwarts House'] == labels.reset_index(drop=True)
    return matches.mean() * 100


def main() -> None:
    """ Entrypoint to parse arguments and run the service """
    args = parse_arguments()

    logging.info("Loading and preprocessing data...")
    features, labels = load_and_preprocess_data(args.d)
    logging.info("Data loaded successfully.")

    config = TrainingConfig()

    logging.info(f"Starting training using {args.a}...")
    weights = train(args.a, features, labels, config)
    acc = compute_accuracy(weights, features, labels)
    logging.info(f"Training completed. Accuracy: {acc:.3f}%")

    weights.to_pickle('weights.pkl')
    logging.info("Model weights saved to 'weights.pkl'.")


if __name__ == '__main__':
    main()
