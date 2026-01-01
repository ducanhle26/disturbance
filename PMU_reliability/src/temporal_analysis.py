"""
Temporal Analysis Module for PMU Reliability Framework.

Detects anomalies, patterns, and clustering in disturbance event timing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats

try:
    from .data_loader import _find_datetime_column
except ImportError:
    from data_loader import _find_datetime_column


class TemporalAnalyzer:
    """
    Temporal pattern analyzer for PMU disturbance events.
    
    Provides methods for:
    - Anomaly detection (IQR, Z-score methods)
    - Hourly/daily/monthly pattern analysis
    - Peak period identification
    - Temporal clustering tests
    - Inter-arrival time analysis
    """
    
    def __init__(self, events_df: pd.DataFrame, datetime_col: Optional[str] = None):
        """
        Initialize temporal analyzer.
        
        Parameters
        ----------
        events_df : pd.DataFrame
            Disturbance events dataframe
        datetime_col : str, optional
            Name of datetime column (auto-detected if not provided)
        """
        self.events_df = events_df.copy()
        self.datetime_col = datetime_col or _find_datetime_column(events_df)
        
        if self.datetime_col is None:
            raise ValueError("No datetime column found in events dataframe")
        
        self.events_df = self.events_df.sort_values(self.datetime_col)
    
    def detect_anomalies(self, method: str = 'iqr', 
                         threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect anomalous days with unusually high event counts.
        
        Parameters
        ----------
        method : str
            Detection method: 'iqr' or 'zscore'
        threshold : float
            IQR multiplier (default 1.5) or Z-score threshold (default 3.0)
            
        Returns
        -------
        pd.DataFrame
            Anomalous days with event counts
        """
        daily_counts = self._get_daily_counts()
        
        if method == 'iqr':
            q1 = daily_counts['count'].quantile(0.25)
            q3 = daily_counts['count'].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + threshold * iqr
            anomalies = daily_counts[daily_counts['count'] > upper_bound].copy()
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(daily_counts['count']))
            anomalies = daily_counts[z_scores > threshold].copy()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
        
        anomalies = anomalies.sort_values('count', ascending=False)
        return anomalies
    
    def calculate_hourly_pattern(self) -> pd.Series:
        """
        Calculate event counts by hour of day.
        
        Returns
        -------
        pd.Series
            Hourly event counts (index 0-23)
        """
        hours = self.events_df[self.datetime_col].dt.hour
        hourly_counts = hours.value_counts().sort_index()
        hourly_counts = hourly_counts.reindex(range(24), fill_value=0)
        return hourly_counts
    
    def calculate_daily_pattern(self) -> pd.Series:
        """
        Calculate event counts by day of week.
        
        Returns
        -------
        pd.Series
            Daily event counts (Monday=0, Sunday=6)
        """
        days = self.events_df[self.datetime_col].dt.dayofweek
        daily_counts = days.value_counts().sort_index()
        daily_counts = daily_counts.reindex(range(7), fill_value=0)
        daily_counts.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                              'Friday', 'Saturday', 'Sunday']
        return daily_counts
    
    def calculate_monthly_pattern(self) -> pd.Series:
        """
        Calculate event counts by month.
        
        Returns
        -------
        pd.Series
            Monthly event counts (1-12)
        """
        months = self.events_df[self.datetime_col].dt.month
        monthly_counts = months.value_counts().sort_index()
        monthly_counts = monthly_counts.reindex(range(1, 13), fill_value=0)
        monthly_counts.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        return monthly_counts
    
    def calculate_peak_periods(self) -> Dict[str, Any]:
        """
        Identify peak hour, day, and month.
        
        Returns
        -------
        Dict[str, Any]
            Peak periods with their counts
        """
        hourly = self.calculate_hourly_pattern()
        daily = self.calculate_daily_pattern()
        monthly = self.calculate_monthly_pattern()
        
        return {
            'peak_hour': int(hourly.idxmax()),
            'peak_hour_count': int(hourly.max()),
            'peak_day': daily.idxmax(),
            'peak_day_count': int(daily.max()),
            'peak_month': monthly.idxmax(),
            'peak_month_count': int(monthly.max())
        }
    
    def test_clustering(self) -> Dict[str, Any]:
        """
        Test for temporal clustering using dispersion index.
        
        A dispersion index > 1 indicates clustering (events occur in bursts).
        A dispersion index < 1 indicates regularity.
        A dispersion index ≈ 1 indicates random (Poisson) timing.
        
        Returns
        -------
        Dict[str, Any]
            Clustering test results
        """
        daily_counts = self._get_daily_counts()
        
        mean_count = daily_counts['count'].mean()
        var_count = daily_counts['count'].var()
        
        if mean_count == 0:
            return {
                'dispersion_index': None,
                'is_clustered': None,
                'interpretation': 'No events to analyze'
            }
        
        dispersion_index = var_count / mean_count
        
        is_clustered = dispersion_index > 1.0
        
        if dispersion_index > 1.5:
            interpretation = 'Strong clustering (events occur in bursts)'
        elif dispersion_index > 1.0:
            interpretation = 'Moderate clustering'
        elif dispersion_index > 0.8:
            interpretation = 'Approximately random (Poisson)'
        else:
            interpretation = 'Regular spacing (under-dispersed)'
        
        return {
            'dispersion_index': dispersion_index,
            'is_clustered': is_clustered,
            'mean_daily_count': mean_count,
            'variance': var_count,
            'interpretation': interpretation
        }
    
    def calculate_inter_arrival_times(self) -> pd.Series:
        """
        Calculate inter-arrival times between consecutive events.
        
        Returns
        -------
        pd.Series
            Inter-arrival times in days
        """
        sorted_events = self.events_df.sort_values(self.datetime_col)
        inter_arrival = sorted_events[self.datetime_col].diff().dropna()
        inter_arrival_days = inter_arrival.dt.total_seconds() / 86400
        return inter_arrival_days
    
    def get_inter_arrival_statistics(self) -> Dict[str, float]:
        """
        Get statistics on inter-arrival times.
        
        Returns
        -------
        Dict[str, float]
            Statistics including mean, median, std, min, max
        """
        ia_times = self.calculate_inter_arrival_times()
        
        if len(ia_times) == 0:
            return {'mean': None, 'median': None, 'std': None, 'min': None, 'max': None}
        
        return {
            'mean': ia_times.mean(),
            'median': ia_times.median(),
            'std': ia_times.std(),
            'min': ia_times.min(),
            'max': ia_times.max(),
            'count': len(ia_times)
        }
    
    def _get_daily_counts(self) -> pd.DataFrame:
        """Get event counts by date."""
        dates = self.events_df[self.datetime_col].dt.date
        daily_counts = dates.value_counts().reset_index()
        daily_counts.columns = ['date', 'count']
        daily_counts = daily_counts.sort_values('date')
        return daily_counts
    
    def decompose_time_series(self, period: int = 30) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Parameters
        ----------
        period : int
            Seasonal period in days (default 30 for monthly)
            
        Returns
        -------
        Dict[str, pd.Series]
            Decomposition components
        """
        daily_counts = self._get_daily_counts()
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts = daily_counts.set_index('date')
        
        full_range = pd.date_range(
            start=daily_counts.index.min(),
            end=daily_counts.index.max(),
            freq='D'
        )
        daily_counts = daily_counts.reindex(full_range, fill_value=0)
        daily_counts.columns = ['count']
        
        trend = daily_counts['count'].rolling(window=period, center=True).mean()
        
        detrended = daily_counts['count'] - trend
        seasonal = detrended.groupby(detrended.index.dayofyear % period).transform('mean')
        
        residual = daily_counts['count'] - trend - seasonal
        
        return {
            'observed': daily_counts['count'],
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
