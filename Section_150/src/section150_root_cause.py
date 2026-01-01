"""
Root cause analysis for Section 150.
Statistical tests, top causes, time-to-failure, seasonal patterns.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'EDA', 'src'))

from temporal import (
    decompose_time_series,
    aggregate_disturbances_by_time,
    calculate_inter_arrival_times,
    extract_cyclical_patterns
)
import config_section150 as cfg


def test_failure_rate_significance(section150_events: int,
                                   network_mean: float,
                                   network_total: int,
                                   n_sections: int) -> Dict:
    """
    Test if Section 150's failure rate is significantly higher than network.
    
    Uses Poisson test and rate ratio confidence interval.
    
    Returns:
    --------
    Dict: Statistical test results
    """
    # Poisson exact test (one-sided, testing if rate > expected)
    expected = network_mean
    poisson_p = 1 - stats.poisson.cdf(section150_events - 1, expected)
    
    # Rate ratio with confidence interval
    # Using normal approximation for rate ratio
    rate_ratio = section150_events / network_mean if network_mean > 0 else float('inf')
    
    # Confidence interval for rate ratio (using log transform)
    if section150_events > 0 and network_mean > 0:
        log_rr = np.log(rate_ratio)
        se_log_rr = np.sqrt(1/section150_events + 1/network_mean)
        z = stats.norm.ppf(1 - cfg.ALPHA/2)
        ci_lower = np.exp(log_rr - z * se_log_rr)
        ci_upper = np.exp(log_rr + z * se_log_rr)
    else:
        ci_lower, ci_upper = None, None
    
    # Z-test for proportions
    p_150 = section150_events / network_total
    p_expected = 1 / n_sections
    se = np.sqrt(p_expected * (1 - p_expected) / network_total)
    z_score = (p_150 - p_expected) / se if se > 0 else 0
    z_p_value = 1 - stats.norm.cdf(z_score)
    
    return {
        'section150_events': section150_events,
        'expected_events': network_mean,
        'rate_ratio': rate_ratio,
        'rate_ratio_ci_95': (ci_lower, ci_upper),
        'poisson_p_value': poisson_p,
        'poisson_significant': poisson_p < cfg.ALPHA,
        'z_score': z_score,
        'z_p_value': z_p_value,
        'z_significant': z_p_value < cfg.ALPHA,
        'conclusion': 'Significantly higher' if poisson_p < cfg.ALPHA else 'Not significantly different'
    }


def analyze_top_causes(section150_events: pd.DataFrame,
                       all_disturbances: pd.DataFrame,
                       cause_col: str,
                       n_top: int = 5) -> pd.DataFrame:
    """
    Analyze top causes for Section 150 with relative risk vs network.
    
    Returns:
    --------
    pd.DataFrame: Top causes with statistics
    """
    # Section 150 cause counts
    sec150_causes = section150_events[cause_col].value_counts()
    sec150_total = len(section150_events)
    
    # Network cause counts (excluding Section 150)
    other_sections = all_disturbances[all_disturbances['SectionID'] != cfg.TARGET_SECTION_ID]
    network_causes = other_sections[cause_col].value_counts()
    network_total = len(other_sections)
    
    # Calculate relative risk for each cause
    results = []
    for cause in sec150_causes.head(n_top).index:
        sec150_count = sec150_causes.get(cause, 0)
        network_count = network_causes.get(cause, 0)
        
        # Proportions
        sec150_prop = sec150_count / sec150_total if sec150_total > 0 else 0
        network_prop = network_count / network_total if network_total > 0 else 0
        
        # Relative risk
        relative_risk = sec150_prop / network_prop if network_prop > 0 else float('inf')
        
        # Chi-square test for this specific cause
        # 2x2 contingency: Section150 vs Network, This Cause vs Other Causes
        observed = np.array([
            [sec150_count, sec150_total - sec150_count],
            [network_count, network_total - network_count]
        ])
        
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        except:
            chi2, p_value = None, None
        
        results.append({
            'Cause': cause,
            'Section_150_Count': sec150_count,
            'Section_150_Pct': sec150_prop * 100,
            'Network_Count': network_count,
            'Network_Pct': network_prop * 100,
            'Relative_Risk': relative_risk,
            'Chi2_Statistic': chi2,
            'P_Value': p_value,
            'Significant': p_value < cfg.ALPHA if p_value else None
        })
    
    return pd.DataFrame(results)


def analyze_time_to_failure(section150_events: pd.DataFrame,
                            all_disturbances: pd.DataFrame,
                            datetime_col: str) -> Dict:
    """
    Analyze time between consecutive failures for Section 150 vs network.
    
    Returns:
    --------
    Dict: Time-to-failure analysis
    """
    # Section 150 inter-arrival times
    sec150_sorted = section150_events.sort_values(datetime_col)
    sec150_iat = sec150_sorted[datetime_col].diff().dt.total_seconds() / 3600  # hours
    sec150_iat = sec150_iat.dropna()
    
    # Network inter-arrival times (per section)
    network_iat = []
    for section_id, group in all_disturbances.groupby('SectionID'):
        if section_id == cfg.TARGET_SECTION_ID:
            continue
        sorted_group = group.sort_values(datetime_col)
        iat = sorted_group[datetime_col].diff().dt.total_seconds() / 3600
        network_iat.extend(iat.dropna().values)
    network_iat = pd.Series(network_iat)
    
    # Statistical comparison (Kolmogorov-Smirnov test)
    if len(sec150_iat) > 2 and len(network_iat) > 2:
        ks_stat, ks_p = stats.ks_2samp(sec150_iat, network_iat)
    else:
        ks_stat, ks_p = None, None
    
    # Mann-Whitney U test
    if len(sec150_iat) > 2 and len(network_iat) > 2:
        mw_stat, mw_p = stats.mannwhitneyu(sec150_iat, network_iat, alternative='two-sided')
    else:
        mw_stat, mw_p = None, None
    
    return {
        'section150_iat': sec150_iat,
        'section150_mean_hours': sec150_iat.mean(),
        'section150_median_hours': sec150_iat.median(),
        'section150_std_hours': sec150_iat.std(),
        'section150_min_hours': sec150_iat.min(),
        'section150_max_hours': sec150_iat.max(),
        'network_mean_hours': network_iat.mean(),
        'network_median_hours': network_iat.median(),
        'network_std_hours': network_iat.std(),
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p,
        'ks_conclusion': 'Different distributions' if ks_p and ks_p < cfg.ALPHA else 'Similar distributions',
        'mannwhitney_statistic': mw_stat,
        'mannwhitney_p_value': mw_p,
        'mtbf_ratio': network_iat.mean() / sec150_iat.mean() if sec150_iat.mean() > 0 else None
    }


def analyze_seasonal_patterns(section150_events: pd.DataFrame,
                              datetime_col: str) -> Dict:
    """
    Analyze seasonal and cyclical patterns specific to Section 150.
    
    Returns:
    --------
    Dict: Seasonal pattern analysis
    """
    # Extract cyclical patterns
    patterns = extract_cyclical_patterns(section150_events, datetime_col)
    
    # Daily aggregation for decomposition
    daily = aggregate_disturbances_by_time(section150_events, datetime_col, 'D')
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq='D'), fill_value=0)
    
    # Monthly aggregation
    monthly = aggregate_disturbances_by_time(section150_events, datetime_col, 'ME')
    
    # Yearly aggregation
    yearly = section150_events.groupby(section150_events[datetime_col].dt.year).size()
    
    # Peak detection
    peak_hour = patterns['hourly'].idxmax()
    peak_day = patterns['daily'].idxmax()
    peak_month = patterns['monthly'].idxmax()
    
    # Weekend effect
    weekend_events = patterns['weekend_vs_weekday'].get(True, 0)
    weekday_events = patterns['weekend_vs_weekday'].get(False, 0)
    weekend_ratio = weekend_events / (weekday_events / 5 * 2) if weekday_events > 0 else 0
    
    # Chi-square test for uniform distribution across months
    monthly_counts = patterns['monthly'].values
    expected_monthly = np.full(12, len(section150_events) / 12)
    chi2_monthly, p_monthly = stats.chisquare(monthly_counts, expected_monthly)
    
    # Time series decomposition (if enough data)
    decomposition = None
    if len(daily) > 30:
        try:
            decomposition = decompose_time_series(daily, period=7)
        except Exception as e:
            print(f"Decomposition failed: {e}")
    
    return {
        'hourly_pattern': patterns['hourly'],
        'daily_pattern': patterns['daily'],
        'monthly_pattern': patterns['monthly'],
        'yearly_trend': yearly,
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'peak_month': peak_month,
        'weekend_ratio': weekend_ratio,
        'weekend_effect': 'Higher on weekends' if weekend_ratio > 1.2 else ('Lower on weekends' if weekend_ratio < 0.8 else 'No significant difference'),
        'monthly_chi2': chi2_monthly,
        'monthly_p_value': p_monthly,
        'seasonal_variation': 'Significant' if p_monthly < cfg.ALPHA else 'Not significant',
        'decomposition': decomposition,
        'daily_counts': daily
    }


def get_root_cause_summary(section150_events: pd.DataFrame,
                           all_disturbances: pd.DataFrame,
                           network_baselines: Dict) -> Dict:
    """
    Complete root cause analysis for Section 150.
    
    Returns:
    --------
    Dict: All root cause analyses
    """
    datetime_col = network_baselines['datetime_col']
    cause_col = network_baselines['cause_col']
    
    # Statistical significance test
    significance = test_failure_rate_significance(
        len(section150_events),
        network_baselines['mean_events_per_section'],
        network_baselines['total_events'],
        network_baselines['total_sections']
    )
    
    # Top causes analysis
    top_causes = analyze_top_causes(section150_events, all_disturbances, cause_col)
    
    # Time-to-failure analysis
    ttf = analyze_time_to_failure(section150_events, all_disturbances, datetime_col)
    
    # Seasonal patterns
    seasonal = analyze_seasonal_patterns(section150_events, datetime_col)
    
    return {
        'significance_tests': significance,
        'top_causes': top_causes,
        'time_to_failure': ttf,
        'seasonal_patterns': seasonal
    }
