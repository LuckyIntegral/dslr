import pandas as pd
import numpy as np
import math, sys


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

def main():
    if len(sys.argv) == 1:
        print('Error: missing arguments')
        sys.exit(1)
    if len(sys.argv) > 3:
        print('Error: too many arguments')
        sys.exit(1)
    if len(sys.argv) == 3 and sys.argv[2] != 'bonus':
        print('Error: invalid argument')
        sys.exit(1)
    bonus = len(sys.argv) == 3 and sys.argv[2] == 'bonus'
    try:
        df = pd.read_csv(sys.argv[1])
        print(ft_describe(df, bonus).to_string())
        # print(df.describe().to_string())
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
