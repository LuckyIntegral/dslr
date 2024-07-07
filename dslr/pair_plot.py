from matplotlib import pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
from utils import HOUSE_COLORS


class HogwartsPairPlotService:
    """Service to pair plot data"""
    def __init__(self, path: str) -> None:
        """Initialize the service with the dataset"""
        self.df = pd.read_csv(path)
        self.df.drop(
            inplace=True,
            columns=[
                'Index',
                'First Name',
                'Last Name',
                'Birthday',
                'Best Hand'
            ])

    def plot_data(self) -> None:
        """Plot pair plot for the dataset"""
        sns.pairplot(
            self.df,
            hue='Hogwarts House',
            palette=HOUSE_COLORS
        )
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
