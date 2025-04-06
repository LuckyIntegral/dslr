from argparse import ArgumentParser, Namespace
from typing import List
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DEFAULT_FEATURES_TO_DROP = [
    'Care of Magical Creatures',
    'Arithmancy',
    'Defense Against the Dark Arts',
    'First Name',
    'Last Name',
    'Birthday',
    'Best Hand',
    'Index'
]


def process_data(df: pd.DataFrame,
                 features_to_drop: List[str]
                 ) -> pd.DataFrame:
    """ Drop specified columns and fill missing values """
    df.drop(
        columns=[col for col in features_to_drop if col in df.columns],
        inplace=True,
    )
    df.fillna(0.0, inplace=True)

    return df


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Preprocess a Hogwarts dataset")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to input CSV')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path to save processed CSV')
    parser.add_argument('--drop-cols', type=str, nargs='*',
                        default=DEFAULT_FEATURES_TO_DROP,
                        help='Columns to drop from the dataset')

    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.info("Loading dataset...")
    df = pd.read_csv(args.input)

    logging.info("Processing dataset...")
    processed_df = process_data(df, features_to_drop=args.drop_cols)

    logging.info(f"Saving processed dataset to {args.output}")
    processed_df.to_csv(args.output, index=False)
    logging.info("Done processing dataset.")


if __name__ == '__main__':
    main()
