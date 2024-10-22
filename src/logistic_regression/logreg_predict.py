from argparse import ArgumentParser
from logistic_regression import LogisticRegression

def parse_argument() -> ArgumentParser:
    """Parse command line arguments"""
    parser = ArgumentParser(
        usage='logreg_predict.py -d <path to the data> -m <path to the model>'
    )

    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=True,
                        help='Path to data')

    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Path to model')

    return parser.parse_args()


def main() -> None:
    """Entrypoint to parse arguments and run the service"""
    try:
        args = parse_argument()
        model = LogisticRegression(args.data)
        model.load_model(args.model)
        predictions = model.predict(args.data)
        predictions.to_csv('houses.csv')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
