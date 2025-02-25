from argparse import ArgumentParser
from logistic_regression import LogisticRegression


def parse_argument() -> ArgumentParser:
    """Parse command line arguments"""
    parser = ArgumentParser(
        usage='logreg_train.py -d <path to your data> -a <algorithm>'
    )

    parser.add_argument('-d',
                        '--data',
                        type=str,
                        required=True,
                        help='Path to data')

    parser.add_argument('-a',
                        '--algorithm',
                        type=str,
                        required=False,
                        default='stochastic_gradient_descent',
                        choices=['gradient_descent', 'stochastic_gradient_descent', 'mini_batch_gradient_descent'],
                        help='Algorithm to use for training')

    return parser.parse_args()


def main() -> None:
    """Entrypoint to parse arguments and run the service"""
    try:
        args = parse_argument()
        model = LogisticRegression(args.data)
        model.train(args.algorithm)
        model.save_model('weights.pkl')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
