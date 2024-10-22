from matplotlib import pyplot as plt
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
from constants import HOUSE_COLORS, COURSES, IMAGES_FOLDER


class ScatterPlotService:
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
        sns.scatterplot(
            data=self.df,
            x=self.courses[0],
            y=self.courses[1],
            hue='Hogwarts House',
            palette=HOUSE_COLORS
        )
        plt.title(f"{self.courses[0]} vs {self.courses[1]}")
        plt.tight_layout()
        try:
            plt.savefig(f'{IMAGES_FOLDER}/scatter_plot_{self.courses[0]}_{self.courses[1]}.png')
        except FileNotFoundError:
            print(f'Error: The folder {IMAGES_FOLDER} was not found.')
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
        histogram_service = ScatterPlotService(args.data, args.course)
        histogram_service.plot_data()
    except FileNotFoundError:
        print(f'Error: The file {args.data} was not found.')
    except pd.errors.ParserError:
        print(f'Error: The file {args.data} could not be parsed.')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
