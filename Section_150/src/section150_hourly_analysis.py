"""
Hourly analysis for Section 150.
Deep-dive into the 7 PM peak finding with multi-factor analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_section150 as cfg


def compare_hourly_distributions(section150_events: pd.DataFrame,
                                  network_events: pd.DataFrame,
                                  datetime_col: str) -> Dict:
    """
    Compare Section 150 vs network hourly event distribution.

    Statistical test: Chi-square goodness-of-fit

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events
    network_events : pd.DataFrame
        All network events
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    Dict with:
        - section150_hourly: Series (24 hours)
        - network_hourly: Series (24 hours, normalized)
        - section150_hourly_pct: Series (percentages)
        - network_hourly_pct: Series (percentages)
        - peak_hour_150: int
        - peak_hour_network: int
        - hour_19_count_150: int
        - hour_19_pct_150: float
        - hour_19_pct_network: float
        - enrichment_ratio: float (Sec150/Network at hour 19)
        - chi2_statistic: float
        - p_value: float
        - significant_hours: List[int] (hours where Sec 150 differs significantly)
    """
    # Extract hours
    sec150_hours = section150_events[datetime_col].dt.hour
    network_hours = network_events[datetime_col].dt.hour

    # Hourly counts
    sec150_hourly = sec150_hours.value_counts().sort_index()
    network_hourly = network_hours.value_counts().sort_index()

    # Ensure all 24 hours are present
    all_hours = pd.Series(0, index=range(24))
    sec150_hourly = sec150_hourly.add(all_hours, fill_value=0)
    network_hourly = network_hourly.add(all_hours, fill_value=0)

    # Percentages
    sec150_hourly_pct = (sec150_hourly / sec150_hourly.sum() * 100)
    network_hourly_pct = (network_hourly / network_hourly.sum() * 100)

    # Peak hours
    peak_hour_150 = sec150_hourly.idxmax()
    peak_hour_network = network_hourly.idxmax()

    # Hour 19 (7 PM) specifics
    hour_19_count_150 = sec150_hourly.get(19, 0)
    hour_19_pct_150 = sec150_hourly_pct.get(19, 0)
    hour_19_pct_network = network_hourly_pct.get(19, 0)

    # Enrichment ratio (how much more at hour 19)
    enrichment_ratio = hour_19_pct_150 / hour_19_pct_network if hour_19_pct_network > 0 else float('inf')

    # Chi-square test: Is Section 150 hourly distribution different from network?
    observed = sec150_hourly.values
    expected_proportions = network_hourly_pct.values / 100
    expected = expected_proportions * observed.sum()

    try:
        chi2, p_value = stats.chisquare(observed, expected)
    except:
        chi2, p_value = None, None

    # Find significantly different hours (simple z-test for each hour)
    significant_hours = []
    for hour in range(24):
        obs = sec150_hourly.get(hour, 0)
        exp = expected[hour]
        if exp > 5:  # Minimum expected frequency
            z = (obs - exp) / np.sqrt(exp)
            if abs(z) > 1.96:  # 95% confidence
                significant_hours.append(int(hour))

    return {
        'section150_hourly': sec150_hourly,
        'network_hourly': network_hourly,
        'section150_hourly_pct': sec150_hourly_pct,
        'network_hourly_pct': network_hourly_pct,
        'peak_hour_150': int(peak_hour_150),
        'peak_hour_network': int(peak_hour_network),
        'hour_19_count_150': int(hour_19_count_150),
        'hour_19_pct_150': float(hour_19_pct_150),
        'hour_19_pct_network': float(hour_19_pct_network),
        'enrichment_ratio': float(enrichment_ratio),
        'chi2_statistic': float(chi2) if chi2 else None,
        'p_value': float(p_value) if p_value else None,
        'significant_hours': significant_hours,
        'conclusion': f"Section 150 peaks at hour {peak_hour_150} ({hour_19_pct_150:.1f}% of events), {enrichment_ratio:.1f}x higher than network at hour 19"
    }


def analyze_7pm_cause_breakdown(section150_events: pd.DataFrame,
                                 cause_col: str,
                                 datetime_col: str) -> pd.DataFrame:
    """
    Breakdown of event causes specifically at hour 19 (7 PM).

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events
    cause_col : str
        Name of cause column
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    pd.DataFrame with columns:
        - Cause_Category: str
        - Count_At_19: int
        - Count_Other_Hours: int
        - Pct_At_19: float
        - Pct_Other_Hours: float
        - Enrichment_Ratio: float
        - Fisher_P_Value: float
    """
    # Extract hour and cause category
    events = section150_events.copy()
    events['Hour'] = events[datetime_col].dt.hour

    # Parse cause category
    def parse_category(text):
        if pd.isna(text):
            return 'Unknown'
        text = str(text).strip()
        if ' - ' in text:
            return text.split(' - ')[0].strip()
        return text

    events['Cause_Category'] = events[cause_col].apply(parse_category)

    # Events at hour 19 vs other hours
    at_19 = events[events['Hour'] == 19]
    other_hours = events[events['Hour'] != 19]

    # Cause counts
    causes_at_19 = at_19['Cause_Category'].value_counts()
    causes_other = other_hours['Cause_Category'].value_counts()

    # Get all unique causes
    all_causes = set(causes_at_19.index.tolist() + causes_other.index.tolist())

    results = []
    for cause in all_causes:
        count_at_19 = causes_at_19.get(cause, 0)
        count_other = causes_other.get(cause, 0)

        total_at_19 = len(at_19)
        total_other = len(other_hours)

        pct_at_19 = (count_at_19 / total_at_19 * 100) if total_at_19 > 0 else 0
        pct_other = (count_other / total_other * 100) if total_other > 0 else 0

        enrichment = pct_at_19 / pct_other if pct_other > 0 else float('inf')

        # Fisher's exact test for this cause
        if count_at_19 + count_other >= 5:  # Minimum sample size
            contingency = np.array([
                [count_at_19, total_at_19 - count_at_19],
                [count_other, total_other - count_other]
            ])
            try:
                odds_ratio, p_value = stats.fisher_exact(contingency)
            except:
                p_value = None
        else:
            p_value = None

        results.append({
            'Cause_Category': cause,
            'Count_At_19': int(count_at_19),
            'Count_Other_Hours': int(count_other),
            'Pct_At_19': pct_at_19,
            'Pct_Other_Hours': pct_other,
            'Enrichment_Ratio': enrichment,
            'Fisher_P_Value': float(p_value) if p_value else None
        })

    df = pd.DataFrame(results).sort_values('Count_At_19', ascending=False)
    return df


def analyze_operations_by_hour(section150_events: pd.DataFrame,
                                datetime_col: str) -> Dict:
    """
    Analyze if Operations field varies by time of day.

    Focus on hour 19 vs others.

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    Dict with:
        - ops_by_hour: DataFrame (24 rows x Operations value counts)
        - hour_19_ops_mode: float (most common Operations at 7 PM)
        - other_hours_ops_mode: float
        - hour_19_ops_dist: Counter
        - other_hours_ops_dist: Counter
        - chi2_test: Dict
        - conclusion: str
    """
    if 'Operations' not in section150_events.columns:
        return {'error': 'Operations column not found'}

    events = section150_events.copy()
    events['Hour'] = events[datetime_col].dt.hour

    # Operations by hour pivot
    ops_by_hour = events.groupby(['Hour', 'Operations']).size().unstack(fill_value=0)

    # Hour 19 vs other hours
    at_19 = events[events['Hour'] == 19]['Operations']
    other_hours = events[events['Hour'] != 19]['Operations']

    hour_19_ops = at_19.value_counts()
    other_hours_ops = other_hours.value_counts()

    hour_19_mode = hour_19_ops.idxmax() if len(hour_19_ops) > 0 else None
    other_hours_mode = other_hours_ops.idxmax() if len(other_hours_ops) > 0 else None

    # Chi-square test
    all_ops = set(hour_19_ops.index.tolist() + other_hours_ops.index.tolist())
    all_ops = [v for v in all_ops if not pd.isna(v)]  # Remove NaN

    if len(all_ops) >= 2:
        contingency = []
        for ops_val in all_ops:
            contingency.append([
                hour_19_ops.get(ops_val, 0),
                other_hours_ops.get(ops_val, 0)
            ])

        contingency = np.array(contingency).T

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            chi2_result = {
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'dof': int(dof),
                'significant': p_value < 0.05
            }
        except:
            chi2_result = {'error': 'Chi-square test failed'}
    else:
        chi2_result = {'error': 'Insufficient data for chi-square test'}

    # Conclusion
    if 'p_value' in chi2_result:
        if chi2_result['significant']:
            conclusion = f"Operations field differs significantly at hour 19 (mode: {hour_19_mode}) vs other hours (mode: {other_hours_mode})"
        else:
            conclusion = "No significant difference in Operations field by hour"
    else:
        conclusion = "Insufficient data for statistical comparison"

    return {
        'ops_by_hour': ops_by_hour,
        'hour_19_ops_mode': float(hour_19_mode) if hour_19_mode and not pd.isna(hour_19_mode) else None,
        'other_hours_ops_mode': float(other_hours_mode) if other_hours_mode and not pd.isna(other_hours_mode) else None,
        'hour_19_ops_dist': dict(hour_19_ops.to_dict()),
        'other_hours_ops_dist': dict(other_hours_ops.to_dict()),
        'chi2_test': chi2_result,
        'conclusion': conclusion
    }


def analyze_cromwell_by_hour(events_with_cromwell: pd.DataFrame,
                              datetime_col: str) -> Dict:
    """
    Analyze if Cromwell Tap 31 state varies by hour.

    Requires Cromwell_Tap_State column from operational_context module.

    Parameters:
    -----------
    events_with_cromwell : pd.DataFrame
        Events with Cromwell_Tap_State column
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    Dict with:
        - cromwell_by_hour: DataFrame (24 rows x state counts)
        - hour_19_state_distribution: dict
        - other_hours_state_distribution: dict
        - hour_19_most_common_state: str
        - chi2_statistic: float
        - p_value: float
        - conclusion: str
    """
    if 'Cromwell_Tap_State' not in events_with_cromwell.columns:
        return {'error': 'Cromwell_Tap_State column not found - run operational_context analysis first'}

    events = events_with_cromwell.copy()
    events['Hour'] = events[datetime_col].dt.hour

    # Cromwell state by hour pivot
    cromwell_by_hour = events.groupby(['Hour', 'Cromwell_Tap_State']).size().unstack(fill_value=0)

    # Hour 19 vs other hours
    at_19 = events[events['Hour'] == 19]['Cromwell_Tap_State']
    other_hours = events[events['Hour'] != 19]['Cromwell_Tap_State']

    hour_19_states = at_19.value_counts()
    other_hours_states = other_hours.value_counts()

    hour_19_most_common = hour_19_states.idxmax() if len(hour_19_states) > 0 else 'not_mentioned'

    # Chi-square test
    all_states = set(hour_19_states.index.tolist() + other_hours_states.index.tolist())

    if len(all_states) >= 2:
        contingency = []
        for state in all_states:
            contingency.append([
                hour_19_states.get(state, 0),
                other_hours_states.get(state, 0)
            ])

        contingency = np.array(contingency).T

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except:
            chi2, p_value = None, None
    else:
        chi2, p_value = None, None

    # Conclusion
    if p_value and p_value < 0.05:
        conclusion = f"Cromwell Tap state differs significantly at hour 19 (most common: {hour_19_most_common})"
    else:
        conclusion = "No significant difference in Cromwell Tap state by hour"

    return {
        'cromwell_by_hour': cromwell_by_hour,
        'hour_19_state_distribution': dict(hour_19_states.to_dict()),
        'other_hours_state_distribution': dict(other_hours_states.to_dict()),
        'hour_19_most_common_state': hour_19_most_common,
        'chi2_statistic': float(chi2) if chi2 else None,
        'p_value': float(p_value) if p_value else None,
        'conclusion': conclusion
    }


def analyze_unknown_timing_peak(unknown_events: pd.DataFrame,
                                 datetime_col: str,
                                 total_events: int) -> Dict:
    """
    Check if Unknown events cluster at 7 PM (suggesting operational activities).

    Parameters:
    -----------
    unknown_events : pd.DataFrame
        Unknown events only
    datetime_col : str
        Name of datetime column
    total_events : int
        Total Section 150 events (for comparison)

    Returns:
    --------
    Dict with:
        - unknown_hourly_distribution: Series (24 hours)
        - unknown_at_19: int
        - unknown_pct_at_19: float (% of Unknown events at hour 19)
        - all_events_pct_at_19: float (% of all events at hour 19)
        - enrichment_ratio: float
        - binomial_p_value: float
        - conclusion: str
    """
    if len(unknown_events) == 0:
        return {'error': 'No Unknown events'}

    unknown_hours = unknown_events[datetime_col].dt.hour
    unknown_hourly = unknown_hours.value_counts().sort_index()

    # Ensure all 24 hours
    all_hours = pd.Series(0, index=range(24))
    unknown_hourly = unknown_hourly.add(all_hours, fill_value=0)

    unknown_at_19 = unknown_hourly.get(19, 0)
    unknown_pct_at_19 = (unknown_at_19 / len(unknown_events) * 100) if len(unknown_events) > 0 else 0

    # Compare to overall Section 150 distribution
    # Assume this is provided or calculate from context
    # For now, use expected value under uniform distribution
    expected_pct = 100 / 24  # ~4.17% if uniform

    enrichment = unknown_pct_at_19 / expected_pct

    # Binomial test: Is hour 19 over-represented?
    # H0: P(hour=19) = 1/24
    n = len(unknown_events)
    k = unknown_at_19
    p_expected = 1/24

    try:
        # Two-sided test
        from scipy.stats import binom_test
        p_value = binom_test(k, n, p_expected, alternative='greater')
    except:
        p_value = None

    # Conclusion
    if p_value and p_value < 0.05:
        conclusion = f"Unknown events significantly cluster at hour 19 ({unknown_pct_at_19:.1f}% vs expected {expected_pct:.1f}%, p={p_value:.4f}). Suggests operational/switching activities."
    else:
        conclusion = f"Unknown events at hour 19 ({unknown_pct_at_19:.1f}%) not significantly different from expected"

    return {
        'unknown_hourly_distribution': unknown_hourly,
        'unknown_at_19': int(unknown_at_19),
        'unknown_pct_at_19': float(unknown_pct_at_19),
        'expected_pct_uniform': float(expected_pct),
        'enrichment_ratio': float(enrichment),
        'binomial_p_value': float(p_value) if p_value else None,
        'conclusion': conclusion
    }


def analyze_day_hour_interaction(section150_events: pd.DataFrame,
                                  datetime_col: str) -> Dict:
    """
    Analyze if 7 PM peak is consistent across all days or stronger on certain days.

    Creates 7 (days) x 24 (hours) heatmap data.

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    Dict with:
        - day_hour_matrix: DataFrame (7 rows=days, 24 cols=hours)
        - peak_day_at_19: str (day with most events at 7 PM)
        - weekday_19_count: int
        - weekend_19_count: int
        - weekday_other_count: int
        - weekend_other_count: int
        - interaction_chi2: float
        - interaction_p_value: float
        - conclusion: str
    """
    events = section150_events.copy()
    events['Hour'] = events[datetime_col].dt.hour
    events['DayOfWeek'] = events[datetime_col].dt.dayofweek  # 0=Monday, 6=Sunday
    events['DayName'] = events[datetime_col].dt.day_name()

    # Day x Hour matrix
    day_hour_counts = events.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)

    # Ensure all days and hours present
    day_hour_matrix = pd.DataFrame(0, index=range(7), columns=range(24))
    day_hour_matrix.update(day_hour_counts)

    # Add day names as index
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_hour_matrix.index = day_names

    # Peak day at hour 19
    if 19 in day_hour_matrix.columns:
        peak_day_idx = day_hour_matrix[19].idxmax()
        peak_day_at_19 = peak_day_idx
    else:
        peak_day_at_19 = None

    # Weekday vs weekend at hour 19
    at_19 = events[events['Hour'] == 19]
    other_hours = events[events['Hour'] != 19]

    weekday_19 = at_19[at_19['DayOfWeek'] < 5]  # Mon-Fri
    weekend_19 = at_19[at_19['DayOfWeek'] >= 5]  # Sat-Sun

    weekday_other = other_hours[other_hours['DayOfWeek'] < 5]
    weekend_other = other_hours[other_hours['DayOfWeek'] >= 5]

    weekday_19_count = len(weekday_19)
    weekend_19_count = len(weekend_19)
    weekday_other_count = len(weekday_other)
    weekend_other_count = len(weekend_other)

    # Interaction test: Is hour 19 effect different on weekdays vs weekends?
    contingency = np.array([
        [weekday_19_count, weekday_other_count],
        [weekend_19_count, weekend_other_count]
    ])

    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    except:
        chi2, p_value = None, None

    # Conclusion
    if p_value and p_value < 0.05:
        if weekday_19_count > weekend_19_count:
            conclusion = f"7 PM peak significantly stronger on weekdays ({weekday_19_count} events) than weekends ({weekend_19_count} events)"
        else:
            conclusion = f"7 PM peak significantly stronger on weekends ({weekend_19_count} events) than weekdays ({weekday_19_count} events)"
    else:
        conclusion = "7 PM peak consistent across weekdays and weekends"

    return {
        'day_hour_matrix': day_hour_matrix,
        'peak_day_at_19': peak_day_at_19,
        'weekday_19_count': int(weekday_19_count),
        'weekend_19_count': int(weekend_19_count),
        'weekday_other_count': int(weekday_other_count),
        'weekend_other_count': int(weekend_other_count),
        'interaction_chi2': float(chi2) if chi2 else None,
        'interaction_p_value': float(p_value) if p_value else None,
        'conclusion': conclusion
    }


def get_hourly_analysis_summary(section150_events: pd.DataFrame,
                                 disturbance_df: pd.DataFrame,
                                 operational_context: Dict,
                                 unknown_analysis: Dict,
                                 network_baselines: Dict) -> Dict:
    """
    Complete hourly analysis for Section 150.

    Integrates outputs from operational_context and unknown_analysis modules.

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events
    disturbance_df : pd.DataFrame
        All network events
    operational_context : Dict
        Results from operational_context analysis
    unknown_analysis : Dict
        Results from unknown_analysis
    network_baselines : Dict
        Network baseline statistics

    Returns:
    --------
    Dict with keys:
        - hourly_comparison: Dict
        - cause_breakdown_7pm: DataFrame
        - operations_by_hour: Dict
        - cromwell_by_hour: Dict
        - unknown_timing: Dict
        - day_hour_interaction: Dict
    """
    datetime_col = network_baselines['datetime_col']
    cause_col = network_baselines['cause_col']

    # Hourly distribution comparison
    hourly_comparison = compare_hourly_distributions(
        section150_events, disturbance_df, datetime_col
    )

    # 7 PM cause breakdown
    cause_breakdown_7pm = analyze_7pm_cause_breakdown(
        section150_events, cause_col, datetime_col
    )

    # Operations by hour
    operations_by_hour = analyze_operations_by_hour(
        section150_events, datetime_col
    )

    # Cromwell by hour (uses operational_context output)
    events_with_cromwell = operational_context.get('events_with_context', section150_events)
    cromwell_by_hour = analyze_cromwell_by_hour(
        events_with_cromwell, datetime_col
    )

    # Unknown timing peak
    unknown_events = unknown_analysis.get('unknown_events', pd.DataFrame())
    unknown_timing = analyze_unknown_timing_peak(
        unknown_events, datetime_col, len(section150_events)
    )

    # Day-hour interaction
    day_hour_interaction = analyze_day_hour_interaction(
        section150_events, datetime_col
    )

    return {
        'hourly_comparison': hourly_comparison,
        'cause_breakdown_7pm': cause_breakdown_7pm,
        'operations_by_hour': operations_by_hour,
        'cromwell_by_hour': cromwell_by_hour,
        'unknown_timing': unknown_timing,
        'day_hour_interaction': day_hour_interaction
    }
