from matplotlib import pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS, HOUSES, COURSES


class HogwartsPairPlotService:
    """Service to pair plot data"""
    def __init__(self, path: str) -> None:
        """Initialize the service with the dataset"""
        self.df = pd.read_csv(path)
        # self.df = self.df[["Hogwarts House", 'Arithmancy', 'Astronomy', 'Herbology']]
        self.df = self.df.drop(columns=[
            'Index',
            'First Name',
            'Last Name',
            'Birthday',
            'Best Hand'
        ])

    def plot_data(self) -> None:
        def hist_diag(x: pd.Series, color: str, label: str) -> None:
            plt.hist(x, color=color, label=label, alpha=0.4)

        def scatter_plot(x: pd.Series, y: pd.Series, color: str, label: str) -> None:
            plt.scatter(x, y, marker='.', color=color, label=label, alpha=0.4)

        pair_grid = sns.PairGrid(
            self.df,
            hue="Hogwarts House",
            palette=HOUSE_COLORS,
        )
        pair_grid.map_diag(hist_diag)
        pair_grid.map_upper(scatter_plot)
        pair_grid.map_lower(scatter_plot)

        pair_grid.add_legend(title="Hogwarts House", bbox_to_anchor=(1.05, 0.5), loc='center left')

        plt.show()


def parse_argument() -> ArgumentParser:
    """Parse command line arguments"""
    parser = ArgumentParser(
        usage='histogram.py -d <path to your data>'
    )

    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=False,
                        default='data/dataset_train.csv',
                        help='Path to data')

    return parser.parse_args()


def main() -> None:
    """Entrypoint to parse arguments and run the service"""
    try:
        args = parse_argument()
        pair_plt_service = HogwartsPairPlotService(args.data)
        pair_plt_service.plot_data()
    except FileNotFoundError:
        print(f'Error: The file {args.data} was not found.')
    except pd.errors.ParserError:
        print(f'Error: The file {args.data} could not be parsed.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
