import pandas as pd
import numpy as np
import math
import sys


def filter_series(series: pd.Series) -> pd.Series:
    """Returns series object filtered from na values"""
    return pd.Series(filter(lambda x: pd.notna(x), series), dtype=np.float64)


def ft_count(series: pd.Series) -> np.float64:
    """Counts non-NaN values in the Series"""
    return len(filter_series(series))


def ft_mean(series: pd.Series) -> np.float64:
    """Returns mean value from the series"""
    filtered = filter_series(series)

    if len(filtered) == 0:
        return np.nan

    return sum(filtered) / len(filtered)


def ft_std(series: pd.Series) -> np.float64:
    """Return standart deviation of the series"""
    filtered = filter_series(series)
    mean = ft_mean(series)

    if mean is np.nan:
        return np.nan

    variance = sum((value - mean) ** 2 for value in filtered) / (len(filtered) - 1)
    return variance ** 0.5


def ft_min(series: pd.Series) -> np.float64:
    """Returns minimum value from the series"""
    filtered = filter_series(series)

    if len(filtered) == 0:
        return np.nan

    minimum = filtered[0]
    for value in filtered:
        if value < minimum:
            minimum = value

    return minimum


def ft_max(series: pd.Series) -> np.float64:
    """Returns maximum value from the series"""
    filtered = filter_series(series)

    if len(filtered) == 0:
        return np.nan

    maximum = filtered[0]
    for value in filtered:
        if value > maximum:
            maximum = value

    return maximum


def ft_quartile(series: pd.Series, quartile: int) -> np.float64:
    """Returns mean value from the series"""
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


def ft_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Provides a descriptive statistics summary for numerical columns
    Similar to pd.DataFrame.describe().
    """
    numeric_rows = df.select_dtypes(include=(np.number))
    if len(numeric_rows.columns) < 5:
        raise ValueError('not enough numeric rows')

    res = pd.DataFrame(columns=numeric_rows.columns)
    res.loc['count'] = [ft_count(df[col]) for col in res.columns]
    res.loc['mean'] = [ft_mean(df[col]) for col in res.columns]
    res.loc['std'] = [ft_std(df[col]) for col in res.columns]
    res.loc['min'] = [ft_min(df[col]) for col in res.columns]
    res.loc['25%'] = [ft_quartile(df[col], 1) for col in res.columns]
    res.loc['50%'] = [ft_quartile(df[col], 2) for col in res.columns]
    res.loc['75%'] = [ft_quartile(df[col], 3) for col in res.columns]
    res.loc['max'] = [ft_max(df[col]) for col in res.columns]

    return res


def main():
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <path to dataset>')
        sys.exit(1)
    try:
        df = pd.read_csv(sys.argv[1])
        print(ft_describe(df))
        # print(df.describe(), file=open('orig', '+w'))
        # print(ft_describe(df), file=open('copy', '+w'))
    except FileExistsError as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
