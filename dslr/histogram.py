import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import HOUSE_COLORS, HOUSES, COURSES


class HogwartsHistogramService:
    """Service to plot histograms for Hogwarts houses based on course scores"""
    def __init__(self, path: str, course: str) -> None:
        """Initialize the service with the dataset and the course to analyze
        Args:
            path (str): Path to the CSV dataset
            course (str): The course to analyze
        """
        self.course = course
        self.df = pd.read_csv(path)
        self.df = self.df[['Hogwarts House', self.course]]

    def plot_data(self) -> None:
        """Plot histograms for Hogwarts houses based on the course scores"""
        num_cols = 2
        num_rows = 2

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 7))

        for i, house in enumerate(HOUSES):
            stats = self.df[self.df['Hogwarts House'] == house][self.course]
            ax = axes[i // 2, i % 2]

            ax.hist(stats, color=HOUSE_COLORS[house], stacked=True)
            ax.set_xlabel("Score")
            ax.set_ylabel("Quantity")
            ax.set_title(house)

        plt.tight_layout()
        plt.show()


def parse_argument() -> ArgumentParser:
    """Parse command line arguments"""
    parser = ArgumentParser(
        usage='histogram.py --data <path to your data> --course <course to plot>'
    )

    parser.add_argument('--data',
                        type=str,
                        required=False,
                        default='data/dataset_train.csv',
                        help='Path to data')

    parser.add_argument('--course',
                        type=str,
                        required=False,
                        choices=COURSES,
                        default='Care of Magical Creatures',
                        help='Course for which display hists')

    return parser.parse_args()


def main() -> None:
    """Entrypoint to parse arguments and run the service"""
    try:
        args = parse_argument()
        histogram_service = HogwartsHistogramService(args.data, args.course)
        histogram_service.plot_data()
    except FileNotFoundError:
        print(f'Error: The file {args.data} was not found.')
    except pd.errors.ParserError:
        print(f'Error: The file {args.data} could not be parsed.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
