from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS


def pair_plot(df: pd.DataFrame) -> None:
    sns.pairplot(
        df,
        hue='Hogwarts House',
        palette=HOUSE_COLORS
    )
    plt.savefig('pair_plot.png')
    plt.show()


def main() -> None:
    try:
        dataset = pd.read_csv('data/dataset_train.csv')
        pair_plot(dataset)
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
