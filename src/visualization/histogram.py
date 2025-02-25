from matplotlib import pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS, COURSES


def plot_histogram(df: pd.DataFrame, course: str) -> None:
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
    plt.show()


def parse_argument() -> ArgumentParser:
    parser = ArgumentParser(
        usage='histogram.py -c <course to plot>'
    )

    parser.add_argument('-c',
                        type=str,
                        required=False,
                        choices=COURSES,
                        default='Care of Magical Creatures',
                        help='Course for which display hists')

    return parser.parse_args()


def main() -> None:
    try:
        args = parse_argument()
        dataset = pd.read_csv('data/dataset_train.csv')
        plot_histogram(dataset, args.c)
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
