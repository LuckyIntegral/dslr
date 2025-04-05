from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS
from argparse import ArgumentParser


def pair_plot(df: pd.DataFrame, save: bool = False) -> None:
    sns.pairplot(
        df,
        hue='Hogwarts House',
        palette=HOUSE_COLORS
    )
    if save:
        plt.savefig('images/pair_plot.png')
    plt.show()


def parse_argument() -> ArgumentParser:
    parser = ArgumentParser(
        usage='histogram.py -d <dataset to plot>'
    )

    parser.add_argument('-d', type=str, required=False,
                        default='data/raw/dataset_train.csv',
                        help='Path to the dataset')

    parser.add_argument('-s', action='store_true', default=True,
                        help='Save the plot (Optional)')

    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    dataset = pd.read_csv(args.d)
    pair_plot(dataset, args.s)


if __name__ == '__main__':
    main()
