from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
from utils import HOUSE_COLORS, HOUSES, COURSES


class HogwartsScatterPlotService:
    """Service that compares courses using scatter plot"""
    def __init__(self, path: str, courses: list[str]) -> None:
        """Initialize the service with the dataset and the courses to analyze
        Args:
            path (str): Path to the CSV dataset
            courses (list[str]): The courses to compare
        """
        self.courses = courses
        self.df = pd.read_csv(path)
        self.df = self.df[['Hogwarts House'] + self.courses]

    def plot_data(self) -> None:
        """Displays scatter plot based on the courses scores"""
        plt.title(f'Comparission of {self.courses[0]} and {self.courses[1]}')

        for house in HOUSES:
            stat_x = self.df[self.df['Hogwarts House'] == house][self.courses[0]]
            stat_y = self.df[self.df['Hogwarts House'] == house][self.courses[1]]

            plt.scatter(stat_x, stat_y, alpha=0.4, color=HOUSE_COLORS[house])
            plt.xlabel(self.courses[0])
            plt.ylabel(self.courses[1])

        plt.legend(HOUSES, shadow=True)
        plt.show()


def parse_argument() -> ArgumentParser:
    """Parse command line arguments"""
    parser = ArgumentParser(
        usage='histogram.py -d <data> -c <course to plot> <course to plot>'
    )

    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=False,
                        default='data/dataset_train.csv',
                        help='Path to data')

    parser.add_argument('-c',
                        '--course',
                        type=str,
                        required=False,
                        nargs=2,
                        choices=COURSES,
                        default=['Astronomy', 'Defense Against the Dark Arts'],
                        help='Course for which display hists')

    return parser.parse_args()


def main() -> None:
    """Entrypoint to parse arguments and run the service"""
    try:
        args = parse_argument()
        histogram_service = HogwartsScatterPlotService(args.data, args.course)
        histogram_service.plot_data()
    except FileNotFoundError:
        print(f'Error: The file {args.data} was not found.')
    except pd.errors.ParserError:
        print(f'Error: The file {args.data} could not be parsed.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
