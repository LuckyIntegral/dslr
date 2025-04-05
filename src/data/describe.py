import pandas as pd
import numpy as np
import math
from argparse import ArgumentParser


def filter_series(series: pd.Series) -> pd.Series:
    return pd.Series(filter(lambda x: pd.notna(x), series), dtype=np.float64)


def ft_count(series: pd.Series) -> np.float64:
    return len(filter_series(series))


def ft_mean(series: pd.Series) -> np.float64:
    filtered = filter_series(series)

    if len(filtered) == 0:
        return np.nan

    return sum(filtered) / len(filtered)


def ft_std(series: pd.Series) -> np.float64:
    filtered = filter_series(series)
    mean = ft_mean(series)

    if mean is np.nan:
        return np.nan

    if len(filtered) == 1:
        return np.nan

    variance = sum((value - mean) ** 2 for value in filtered) / (len(filtered) - 1)
    return variance ** 0.5


def ft_min(series: pd.Series) -> np.float64:
    filtered = filter_series(series)

    if len(filtered) == 0:
        return np.nan

    minimum = filtered[0]
    for value in filtered:
        if value < minimum:
            minimum = value

    return minimum


def ft_max(series: pd.Series) -> np.float64:
    filtered = filter_series(series)

    if len(filtered) == 0:
        return np.nan

    maximum = filtered[0]
    for value in filtered:
        if value > maximum:
            maximum = value

    return maximum


def ft_quartile(series: pd.Series, quartile: int) -> np.float64:
    filtered = sorted(filter_series(series))

    if len(filtered) == 0:
        return np.nan

    position = quartile * (len(filtered) - 1) / 4
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return filtered[lower]

    left = filtered[lower] * (upper - position)
    right = filtered[upper] * (position - lower)

    return left + right


def ft_describe(df: pd.DataFrame, is_bonus: bool) -> pd.DataFrame:
    numeric_rows = df.dropna(how='all').select_dtypes(include=(np.number))

    if len(numeric_rows.columns) == 0:
        raise ValueError('No numeric columns')

    res = pd.DataFrame(columns=numeric_rows.columns)
    res.loc['count'] = [ft_count(df[col]) for col in res.columns]
    res.loc['mean'] = [ft_mean(df[col]) for col in res.columns]
    res.loc['std'] = [ft_std(df[col]) for col in res.columns]
    res.loc['min'] = [ft_min(df[col]) for col in res.columns]
    res.loc['25%'] = [ft_quartile(df[col], 1) for col in res.columns]
    res.loc['50%'] = [ft_quartile(df[col], 2) for col in res.columns]
    res.loc['75%'] = [ft_quartile(df[col], 3) for col in res.columns]
    res.loc['max'] = [ft_max(df[col]) for col in res.columns]
    if is_bonus:
        res.loc['range'] = res.loc['max'] - res.loc['min']
        # interquartile range
        res.loc['iqr'] = res.loc['75%'] - res.loc['25%']
        # Measures the asymmetry of the distribution
        res.loc['skewness'] = (res.loc['mean'] - res.loc['50%']) / res.loc['std']

    return res


def parse_argument() -> ArgumentParser:
    parser = ArgumentParser(
        usage='describe.py -d <dataset to plot> -b'
    )

    parser.add_argument('-d',
                        type=str,
                        required=False,
                        default='data/raw/dataset_train.csv',
                        help='Path to the dataset')

    parser.add_argument('-b',
                        action='store_true',
                        help='Bonus argument')

    return parser.parse_args()


def main():
    args = parse_argument()
    df = pd.read_csv(args.d)
    print(ft_describe(df, args.b).to_string())


if __name__ == '__main__':
    main()
