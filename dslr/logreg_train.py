from argparse import ArgumentParser
from logistic_regression import LogisticRegression


def parse_argument() -> ArgumentParser:
    """Parse command line arguments"""
    parser = ArgumentParser(
        usage='logreg_train.py -d <path to your data>'
    )

    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=True,
                        help='Path to data')

    return parser.parse_args()


def main() -> None:
    """Entrypoint to parse arguments and run the service"""
    # try:
    args = parse_argument()
    model = LogisticRegression(args.data)
    model.train()
    model.save_model('weights.csv')
    # except Exception as e:
        # print(f'Error: {e}')


if __name__ == '__main__':
    main()
