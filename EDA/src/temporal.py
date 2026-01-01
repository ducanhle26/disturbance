"""
Temporal analysis utilities for PMU disturbance analysis.
Includes time series decomposition, anomaly detection, change point detection, and pattern analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
import ruptures as rpt
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def aggregate_disturbances_by_time(df: pd.DataFrame,
                                   datetime_col: str,
                                   freq: str = 'D') -> pd.Series:
    """
    Aggregate disturbance counts by time period.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    datetime_col : str
        Name of datetime column
    freq : str
        Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)

    Returns:
    --------
    pd.Series
        Time series of disturbance counts
    """
    df = df.copy()
    df = df.set_index(datetime_col)
    counts = df.resample(freq).size()
    return counts


def decompose_time_series(ts: pd.Series,
                          period: Optional[int] = None,
                          seasonal: int = 7) -> Dict:
    """
    Perform STL (Seasonal-Trend decomposition using LOESS) on time series.

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    period : int, optional
        Seasonal period (auto-detected if None)
    seasonal : int
        Length of seasonal smoother

    Returns:
    --------
    Dict
        Dictionary containing trend, seasonal, and residual components
    """
    # Handle missing values
    ts_clean = ts.fillna(ts.median())

    # Auto-detect period if not provided
    if period is None:
        period = 7 if ts.index.freq == 'D' else 12

    # Perform STL decomposition
    stl = STL(ts_clean, seasonal=seasonal, period=period)
    result = stl.fit()

    return {
        'observed': ts_clean,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid
    }


def detect_anomalies_zscore(ts: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect anomalies using Z-score method.

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    threshold : float
        Z-score threshold (default: 3.0)

    Returns:
    --------
    pd.Series
        Boolean series indicating anomalies
    """
    z_scores = np.abs(stats.zscore(ts.fillna(ts.median())))
    anomalies = z_scores > threshold
    return pd.Series(anomalies, index=ts.index)


def detect_anomalies_iqr(ts: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect anomalies using IQR (Interquartile Range) method.

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    multiplier : float
        IQR multiplier (default: 1.5)

    Returns:
    --------
    pd.Series
        Boolean series indicating anomalies
    """
    Q1 = ts.quantile(0.25)
    Q3 = ts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    anomalies = (ts < lower_bound) | (ts > upper_bound)
    return anomalies


def detect_anomalies_isolation_forest(ts: pd.Series,
                                      contamination: float = 0.1) -> pd.Series:
    """
    Detect anomalies using Isolation Forest algorithm.

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    contamination : float
        Expected proportion of outliers (default: 0.1)

    Returns:
    --------
    pd.Series
        Boolean series indicating anomalies
    """
    # Reshape for sklearn
    X = ts.values.reshape(-1, 1)

    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X)

    # Convert to boolean (1 = normal, -1 = anomaly)
    anomalies = predictions == -1
    return pd.Series(anomalies, index=ts.index)


def calculate_inter_arrival_times(df: pd.DataFrame,
                                  datetime_col: str,
                                  group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate time between consecutive disturbances.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    datetime_col : str
        Name of datetime column
    group_by : str, optional
        Column to group by (e.g., 'SectionID')

    Returns:
    --------
    pd.DataFrame
        Inter-arrival times in hours
    """
    df = df.sort_values(datetime_col)

    if group_by:
        # Calculate inter-arrival times per group
        df['inter_arrival_hours'] = df.groupby(group_by)[datetime_col].diff().dt.total_seconds() / 3600
    else:
        # Calculate overall inter-arrival times
        df['inter_arrival_hours'] = df[datetime_col].diff().dt.total_seconds() / 3600

    return df


def detect_change_points(ts: pd.Series,
                         model: str = 'rbf',
                         min_size: int = 2,
                         jump: int = 5,
                         pen: float = 3) -> List[int]:
    """
    Detect change points using PELT algorithm.

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    model : str
        Cost function model ('rbf', 'l1', 'l2', 'normal')
    min_size : int
        Minimum segment size
    jump : int
        Subsample parameter
    pen : float
        Penalty value

    Returns:
    --------
    List[int]
        Indices of change points
    """
    # Prepare signal
    signal = ts.fillna(ts.median()).values

    # Detect change points using PELT
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
    change_points = algo.predict(pen=pen)

    # Remove last point (end of series)
    if change_points and change_points[-1] == len(signal):
        change_points = change_points[:-1]

    return change_points


def calculate_rolling_statistics(ts: pd.Series,
                                 windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
    """
    Calculate rolling mean and std for multiple window sizes.

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    windows : List[int]
        List of window sizes in days

    Returns:
    --------
    pd.DataFrame
        Rolling statistics for each window
    """
    result = pd.DataFrame(index=ts.index)
    result['original'] = ts

    for window in windows:
        result[f'rolling_mean_{window}d'] = ts.rolling(window=window, min_periods=1).mean()
        result[f'rolling_std_{window}d'] = ts.rolling(window=window, min_periods=1).std()

    return result


def extract_cyclical_patterns(df: pd.DataFrame,
                              datetime_col: str) -> pd.DataFrame:
    """
    Extract hour-of-day, day-of-week, and monthly patterns.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    pd.DataFrame
        Aggregated patterns
    """
    df = df.copy()
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['day_name'] = df[datetime_col].dt.day_name()
    df['month'] = df[datetime_col].dt.month
    df['month_name'] = df[datetime_col].dt.month_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    patterns = {
        'hourly': df.groupby('hour').size(),
        'daily': df.groupby('day_name').size().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]),
        'monthly': df.groupby('month_name').size().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]),
        'weekend_vs_weekday': df.groupby('is_weekend').size()
    }

    return patterns


def calculate_acf_pacf(ts: pd.Series,
                       nlags: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    nlags : int
        Number of lags

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (acf_values, pacf_values)
    """
    ts_clean = ts.dropna()

    acf_values = acf(ts_clean, nlags=nlags)
    pacf_values = pacf(ts_clean, nlags=nlags)

    return acf_values, pacf_values


def test_poisson_process(inter_arrival_times: pd.Series,
                         alpha: float = 0.05) -> Dict:
    """
    Test if inter-arrival times follow exponential distribution (Poisson process).

    Parameters:
    -----------
    inter_arrival_times : pd.Series
        Inter-arrival times in hours
    alpha : float
        Significance level

    Returns:
    --------
    Dict
        Test results including statistic, p-value, and conclusion
    """
    # Remove NaN values
    times = inter_arrival_times.dropna()

    if len(times) < 3:
        return {'error': 'Insufficient data for testing'}

    # Kolmogorov-Smirnov test for exponential distribution
    # Fit exponential distribution (parameter = 1/mean)
    rate = 1 / times.mean()
    statistic, p_value = stats.kstest(times, lambda x: stats.expon.cdf(x, scale=1/rate))

    result = {
        'statistic': statistic,
        'p_value': p_value,
        'is_poisson': p_value > alpha,
        'mean_inter_arrival': times.mean(),
        'std_inter_arrival': times.std(),
        'conclusion': 'Follows Poisson process' if p_value > alpha else 'Does not follow Poisson process'
    }

    return result
