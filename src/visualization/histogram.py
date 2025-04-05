from matplotlib import pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS, COURSES


def plot_histogram(df: pd.DataFrame, course: str, save: bool = False) -> None:
    sns.histplot(
        df,
        x=course,
        hue='Hogwarts House',
        palette=HOUSE_COLORS,
        element='step',
        multiple='layer'
    )

    plt.title(f"{course} distribution")
    plt.xlabel("Score")
    plt.ylabel("Quantity")
    plt.tight_layout()
    if save:
        plt.savefig(f'images/histogram_{course}.png')
    plt.show()


def parse_argument() -> ArgumentParser:
    parser = ArgumentParser(
        usage='histogram.py -c <course to plot> -d <dataset to plot>'
    )
    parser.add_argument('-c', type=str, required=False, choices=COURSES,
                        default='Care of Magical Creatures',
                        help='Course for which display hists')
    parser.add_argument('-d', type=str, required=False,
                        default='data/raw/dataset_train.csv',
                        help='Path to the dataset')
    parser.add_argument('-s', action='store_true', default=True,
                        help='Save the plot (Optional)')

    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    dataset = pd.read_csv(args.d)
    plot_histogram(dataset, args.c, args.s)


if __name__ == '__main__':
    main()
