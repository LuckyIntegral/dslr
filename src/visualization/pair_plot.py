from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS, PATH_DATASET


def pair_plot(df: pd.DataFrame) -> None:
    sns.pairplot(
        df,
        hue='Hogwarts House',
        palette=HOUSE_COLORS
    )
    plt.savefig('images/pair_plot.png')
    plt.show()


def main() -> None:
    dataset = pd.read_csv(PATH_DATASET)
    pair_plot(dataset)


if __name__ == '__main__':
    main()
