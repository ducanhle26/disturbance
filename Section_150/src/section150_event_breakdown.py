"""
Event breakdown analysis for Section 150.
Timeline, temporal clustering, cause distribution, and network comparison.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'EDA', 'src'))

from temporal import (
    aggregate_disturbances_by_time,
    calculate_inter_arrival_times,
    detect_change_points,
    calculate_rolling_statistics,
    extract_cyclical_patterns,
    test_poisson_process
)
import config_section150 as cfg


def create_event_timeline(section150_events: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Create complete timeline of Section 150 events with detailed information.
    
    Returns:
    --------
    pd.DataFrame: Timeline with all events sorted chronologically
    """
    timeline = section150_events.copy()
    timeline = timeline.sort_values(datetime_col)
    
    # Add sequential event number
    timeline['Event_Number'] = range(1, len(timeline) + 1)
    
    # Add time since previous event
    timeline['Hours_Since_Previous'] = timeline[datetime_col].diff().dt.total_seconds() / 3600
    timeline['Days_Since_Previous'] = timeline['Hours_Since_Previous'] / 24
    
    # Add temporal features
    timeline['Year'] = timeline[datetime_col].dt.year
    timeline['Month'] = timeline[datetime_col].dt.month
    timeline['DayOfWeek'] = timeline[datetime_col].dt.day_name()
    timeline['Hour'] = timeline[datetime_col].dt.hour
    
    return timeline


def analyze_temporal_clustering(section150_events: pd.DataFrame, 
                                datetime_col: str,
                                window_days: int = 7) -> Dict:
    """
    Analyze if events are temporally clustered (bunched) or random.
    
    Returns:
    --------
    Dict: Clustering analysis results
    """
    events = section150_events.copy().sort_values(datetime_col)
    
    # Calculate inter-arrival times
    inter_arrival = events[datetime_col].diff().dt.total_seconds() / 3600  # hours
    inter_arrival = inter_arrival.dropna()
    
    # Test for Poisson process (random arrivals)
    poisson_test = test_poisson_process(inter_arrival)
    
    # Rolling window analysis for burst detection
    daily_counts = aggregate_disturbances_by_time(events, datetime_col, 'D')
    daily_counts = daily_counts.reindex(
        pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq='D'),
        fill_value=0
    )
    
    # Calculate z-scores for burst detection
    rolling_mean = daily_counts.rolling(window=window_days, min_periods=1).mean()
    rolling_std = daily_counts.rolling(window=window_days, min_periods=1).std().fillna(1)
    z_scores = (daily_counts - rolling_mean) / rolling_std.replace(0, 1)
    
    # Identify burst days
    burst_threshold = cfg.BURST_THRESHOLD_ZSCORE
    burst_days = daily_counts[z_scores > burst_threshold]
    
    # Dispersion index (variance/mean ratio) - >1 indicates clustering
    dispersion_index = daily_counts.var() / daily_counts.mean() if daily_counts.mean() > 0 else 0
    
    # Change point detection
    try:
        change_points = detect_change_points(daily_counts, pen=3)
        change_point_dates = [daily_counts.index[cp] for cp in change_points if cp < len(daily_counts)]
    except Exception as e:
        print(f"Change point detection failed: {e}")
        change_points = []
        change_point_dates = []
    
    results = {
        'inter_arrival_mean_hours': inter_arrival.mean(),
        'inter_arrival_median_hours': inter_arrival.median(),
        'inter_arrival_std_hours': inter_arrival.std(),
        'is_poisson_process': poisson_test.get('is_poisson', False),
        'poisson_p_value': poisson_test.get('p_value', None),
        'dispersion_index': dispersion_index,
        'clustering_conclusion': 'Clustered' if dispersion_index > 1.5 else ('Random' if poisson_test.get('is_poisson', False) else 'Slightly Clustered'),
        'burst_days': burst_days,
        'n_burst_days': len(burst_days),
        'change_points': change_points,
        'change_point_dates': change_point_dates,
        'daily_counts': daily_counts,
        'z_scores': z_scores
    }
    
    return results


def analyze_cause_distribution(section150_events: pd.DataFrame,
                               network_cause_dist: pd.Series,
                               cause_col: str) -> Dict:
    """
    Analyze cause distribution for Section 150 vs network average.
    
    Returns:
    --------
    Dict: Cause analysis with comparison to network
    """
    # Section 150 cause distribution
    sec150_causes = section150_events[cause_col].value_counts()
    sec150_causes_pct = section150_events[cause_col].value_counts(normalize=True)
    
    # Top 5 causes
    top5_causes = sec150_causes.head(5)
    
    # Compare to network
    comparison = pd.DataFrame({
        'Section_150_Count': sec150_causes,
        'Section_150_Pct': sec150_causes_pct * 100,
        'Network_Pct': network_cause_dist * 100
    }).fillna(0)
    
    comparison['Difference_Pct'] = comparison['Section_150_Pct'] - comparison['Network_Pct']
    comparison['Ratio'] = comparison['Section_150_Pct'] / comparison['Network_Pct'].replace(0, 0.01)
    
    # Chi-square test for distribution difference
    # Align categories
    all_causes = network_cause_dist.index.tolist()
    observed = sec150_causes.reindex(all_causes, fill_value=0).values
    expected_props = network_cause_dist.reindex(all_causes, fill_value=0).values
    
    # Scale expected to match observed total
    observed_total = observed.sum()
    expected = expected_props * observed_total
    
    # Filter out zero-expected categories
    mask = expected > 0.5  # Minimum expected frequency
    if mask.sum() >= 2:
        try:
            chi2_stat, chi2_p = stats.chisquare(observed[mask], expected[mask])
        except Exception:
            chi2_stat, chi2_p = None, None
    else:
        chi2_stat, chi2_p = None, None
    
    results = {
        'section150_causes': sec150_causes,
        'section150_causes_pct': sec150_causes_pct,
        'top5_causes': top5_causes,
        'comparison_df': comparison,
        'chi2_statistic': chi2_stat,
        'chi2_p_value': chi2_p,
        'distribution_differs': chi2_p < cfg.ALPHA if chi2_p else None
    }
    
    return results


def compare_to_network(section150_events: pd.DataFrame,
                       network_baselines: Dict) -> Dict:
    """
    Compare Section 150 statistics to network averages.
    
    Returns:
    --------
    Dict: Comparison results with statistical tests
    """
    n_events_150 = len(section150_events)
    mean_events = network_baselines['mean_events_per_section']
    std_events = network_baselines['std_events_per_section']
    
    # Z-score: how many standard deviations above/below mean
    z_score = (n_events_150 - mean_events) / std_events if std_events > 0 else 0
    
    # Percentile rank
    events_per_section = network_baselines['events_per_section']
    percentile = stats.percentileofscore(events_per_section, n_events_150)
    
    # Rate ratio
    rate_ratio = n_events_150 / mean_events if mean_events > 0 else float('inf')
    
    # Poisson test: is Section 150's rate significantly higher?
    # Under Poisson assumption, test if observed >> expected
    expected = mean_events
    poisson_p_value = 1 - stats.poisson.cdf(n_events_150 - 1, expected) if expected > 0 else None
    
    results = {
        'section150_events': n_events_150,
        'network_mean': mean_events,
        'network_median': network_baselines['median_events_per_section'],
        'network_std': std_events,
        'network_max': network_baselines['max_events_per_section'],
        'z_score': z_score,
        'percentile_rank': percentile,
        'rate_ratio': rate_ratio,
        'rate_ratio_interpretation': f'Section 150 has {rate_ratio:.1f}x the average event rate',
        'poisson_p_value': poisson_p_value,
        'significantly_higher': poisson_p_value < cfg.ALPHA if poisson_p_value else None
    }
    
    return results


def get_event_breakdown_summary(section150_events: pd.DataFrame,
                                disturbance_df: pd.DataFrame,
                                network_baselines: Dict) -> Dict:
    """
    Complete event breakdown analysis for Section 150.
    
    Returns:
    --------
    Dict: All event breakdown analyses
    """
    datetime_col = network_baselines['datetime_col']
    cause_col = network_baselines['cause_col']
    
    # Timeline
    timeline = create_event_timeline(section150_events, datetime_col)
    
    # Temporal clustering
    clustering = analyze_temporal_clustering(section150_events, datetime_col)
    
    # Cause distribution
    causes = analyze_cause_distribution(
        section150_events, 
        network_baselines['cause_distribution'],
        cause_col
    )
    
    # Network comparison
    comparison = compare_to_network(section150_events, network_baselines)
    
    # Cyclical patterns
    patterns = extract_cyclical_patterns(section150_events, datetime_col)
    
    return {
        'timeline': timeline,
        'clustering': clustering,
        'causes': causes,
        'network_comparison': comparison,
        'cyclical_patterns': patterns,
        'datetime_col': datetime_col,
        'cause_col': cause_col
    }
