from matplotlib import pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS, COURSES, PATH_DATASET


def scatter_plot(df: pd.DataFrame, courses: list) -> None:
    sns.scatterplot(
        data=df,
        x=courses[0],
        y=courses[1],
        hue='Hogwarts House',
        palette=HOUSE_COLORS
    )

    plt.title(f"{courses[0]} vs {courses[1]}")
    plt.tight_layout()
    plt.savefig(f'images/scatter_plot_{courses[0]}_{courses[1]}.png')
    plt.show()


def parse_argument() -> ArgumentParser:
    parser = ArgumentParser(usage='histogram.py -c <course to plot> <course to plot>')
    parser.add_argument('-c', type=str, required=False, nargs=2, choices=COURSES,
                        default=['Astronomy', 'Defense Against the Dark Arts'],
                        help='Course for which display hists')

    return parser.parse_args()


def main() -> None:
    args = parse_argument()
    dataset = pd.read_csv(PATH_DATASET)
    scatter_plot(dataset, args.c)


if __name__ == '__main__':
    main()
