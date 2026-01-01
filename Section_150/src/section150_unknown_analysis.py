"""
Unknown events deep-dive analysis for Section 150.
Forensic analysis of 51 Unknown cause events to find reclassification patterns.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
from collections import Counter
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_section150 as cfg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'EDA', 'src'))

from temporal import (
    calculate_inter_arrival_times,
    detect_change_points,
    test_poisson_process
)


def filter_unknown_events(section150_events: pd.DataFrame,
                          cause_col: str) -> pd.DataFrame:
    """
    Filter to Unknown cause events only.

    Handles variations:
    - "Unknown"
    - "Unknown - ..."

    Parameters:
    -----------
    section150_events : pd.DataFrame
        All Section 150 events
    cause_col : str
        Name of cause column

    Returns:
    --------
    pd.DataFrame: Subset of events with Unknown cause
    """
    # Filter for events starting with "Unknown"
    unknown_mask = section150_events[cause_col].str.startswith('Unknown', na=False)
    unknown_events = section150_events[unknown_mask].copy()

    return unknown_events


def analyze_unknown_temporal_clustering(unknown_events: pd.DataFrame,
                                         datetime_col: str) -> Dict:
    """
    Apply temporal clustering analysis to Unknown subset.

    Reuses existing temporal functions from temporal.py

    Parameters:
    -----------
    unknown_events : pd.DataFrame
        Unknown events only
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    Dict with:
        - count: int
        - mean_iat_hours: float
        - median_iat_hours: float
        - dispersion_index: float (>1 = clustered)
        - burst_days: List[date]
        - poisson_test_result: Dict
        - clustering_conclusion: str
    """
    if len(unknown_events) < 2:
        return {
            'count': len(unknown_events),
            'error': 'Too few events for clustering analysis'
        }

    # Calculate inter-arrival times
    unknown_sorted = unknown_events.sort_values(datetime_col)
    iat = unknown_sorted[datetime_col].diff().dt.total_seconds() / 3600  # hours
    iat = iat.dropna()

    mean_iat = iat.mean()
    median_iat = iat.median()

    # Test for Poisson process
    poisson_test = test_poisson_process(iat)

    # Daily counts for dispersion index
    daily_counts = unknown_events.groupby(
        unknown_events[datetime_col].dt.date
    ).size()

    # Dispersion index (variance/mean ratio) - >1 indicates clustering
    dispersion_index = daily_counts.var() / daily_counts.mean() if daily_counts.mean() > 0 else 0

    # Burst detection (simple threshold: days with >1 Unknown event)
    burst_days = daily_counts[daily_counts > 1].index.tolist()

    # Clustering conclusion
    is_poisson = poisson_test.get('is_poisson', False)
    if dispersion_index > 1.5:
        conclusion = "Strongly clustered"
    elif dispersion_index > 1.0:
        conclusion = "Moderately clustered"
    elif is_poisson:
        conclusion = "Random (Poisson process)"
    else:
        conclusion = "Slightly clustered"

    return {
        'count': len(unknown_events),
        'mean_iat_hours': float(mean_iat),
        'median_iat_hours': float(median_iat),
        'dispersion_index': float(dispersion_index),
        'burst_days': burst_days,
        'n_burst_days': len(burst_days),
        'poisson_test_result': poisson_test,
        'clustering_conclusion': conclusion
    }


def find_time_lagged_correlations(unknown_events: pd.DataFrame,
                                   known_events: pd.DataFrame,
                                   datetime_col: str,
                                   lag_hours: int = 48) -> pd.DataFrame:
    """
    For each Unknown event, find nearest known-cause event within ±lag window.

    Algorithm:
    1. For each Unknown event at time T
    2. Search window [T - lag_hours, T + lag_hours]
    3. Find all known events in window
    4. Calculate time delta to nearest event
    5. Record cause of nearest event

    Parameters:
    -----------
    unknown_events : pd.DataFrame
        Unknown events
    known_events : pd.DataFrame
        Known-cause events (non-Unknown)
    datetime_col : str
        Name of datetime column
    lag_hours : int
        Time window in hours (±lag)

    Returns:
    --------
    pd.DataFrame with columns:
        - Unknown_Event_Time: Timestamp
        - Nearest_Known_Time: Timestamp
        - Time_Delta_Hours: float
        - Nearest_Known_Cause: str
        - Events_In_Window: int
        - Potential_Related_Cause: str
    """
    results = []

    lag_timedelta = pd.Timedelta(hours=lag_hours)

    for idx, unknown_row in unknown_events.iterrows():
        unknown_time = unknown_row[datetime_col]

        # Define search window
        window_start = unknown_time - lag_timedelta
        window_end = unknown_time + lag_timedelta

        # Find known events in window
        in_window = known_events[
            (known_events[datetime_col] >= window_start) &
            (known_events[datetime_col] <= window_end)
        ]

        if len(in_window) == 0:
            results.append({
                'Unknown_Event_Time': unknown_time,
                'Unknown_Event_Index': idx,
                'Nearest_Known_Time': None,
                'Time_Delta_Hours': None,
                'Nearest_Known_Cause': None,
                'Events_In_Window': 0,
                'Potential_Related_Cause': None
            })
            continue

        # Calculate time deltas
        time_deltas = (in_window[datetime_col] - unknown_time).abs()
        nearest_idx = time_deltas.idxmin()
        nearest_event = in_window.loc[nearest_idx]

        nearest_time = nearest_event[datetime_col]
        delta_hours = (nearest_time - unknown_time).total_seconds() / 3600

        # Get cause from known event (parse category)
        cause_text = nearest_event.get('Cause_Category', nearest_event.get('Cause', 'Unknown'))
        if ' - ' in str(cause_text):
            cause_category = cause_text.split(' - ')[0].strip()
        else:
            cause_category = str(cause_text).strip()

        results.append({
            'Unknown_Event_Time': unknown_time,
            'Unknown_Event_Index': idx,
            'Nearest_Known_Time': nearest_time,
            'Time_Delta_Hours': delta_hours,
            'Nearest_Known_Cause': cause_category,
            'Events_In_Window': len(in_window),
            'Potential_Related_Cause': cause_category if abs(delta_hours) < 24 else None
        })

    return pd.DataFrame(results)


def analyze_unknown_operations_field(unknown_events: pd.DataFrame,
                                      all_events: pd.DataFrame) -> Dict:
    """
    Compare Operations field distribution for Unknown vs Known events.

    Statistical test: Chi-square for independence

    Parameters:
    -----------
    unknown_events : pd.DataFrame
        Unknown events
    all_events : pd.DataFrame
        All Section 150 events

    Returns:
    --------
    Dict with:
        - unknown_ops_distribution: Counter
        - known_ops_distribution: Counter
        - chi2_statistic: float
        - p_value: float
        - conclusion: str
        - most_common_unknown_ops: float
        - most_common_known_ops: float
    """
    if 'Operations' not in all_events.columns:
        return {'error': 'Operations column not found'}

    # Get known events (non-Unknown)
    known_events = all_events[~all_events.index.isin(unknown_events.index)]

    # Operations distribution
    unknown_ops = unknown_events['Operations'].value_counts()
    known_ops = known_events['Operations'].value_counts()

    unknown_ops_dist = Counter(unknown_ops.to_dict())
    known_ops_dist = Counter(known_ops.to_dict())

    # Chi-square test
    # Create contingency table
    all_ops_values = set(unknown_ops.index.tolist() + known_ops.index.tolist())

    # Filter out NaN for statistical test
    all_ops_values = [v for v in all_ops_values if not pd.isna(v)]

    if len(all_ops_values) >= 2:
        contingency = []
        for ops_val in all_ops_values:
            unknown_count = unknown_ops.get(ops_val, 0)
            known_count = known_ops.get(ops_val, 0)
            contingency.append([unknown_count, known_count])

        contingency = np.array(contingency).T  # 2 rows (Unknown, Known) x N cols (ops values)

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except:
            chi2, p_value = None, None
    else:
        chi2, p_value = None, None

    # Most common operations values
    most_common_unknown = unknown_ops.idxmax() if len(unknown_ops) > 0 else None
    most_common_known = known_ops.idxmax() if len(known_ops) > 0 else None

    # Conclusion
    if p_value and p_value < 0.05:
        conclusion = f"Unknown events have significantly different Operations distribution (p={p_value:.4f})"
    else:
        conclusion = "No significant difference in Operations field between Unknown and Known events"

    return {
        'unknown_ops_distribution': unknown_ops_dist,
        'known_ops_distribution': known_ops_dist,
        'chi2_statistic': float(chi2) if chi2 else None,
        'p_value': float(p_value) if p_value else None,
        'conclusion': conclusion,
        'most_common_unknown_ops': float(most_common_unknown) if most_common_unknown and not pd.isna(most_common_unknown) else None,
        'most_common_known_ops': float(most_common_known) if most_common_known and not pd.isna(most_common_known) else None
    }


def extract_unknown_description_clues(unknown_events: pd.DataFrame,
                                       cause_col: str) -> pd.DataFrame:
    """
    Text-mine Unknown event descriptions for hidden clues.

    Pattern categories extracted:
    1. Weather mentions: "weather in area", "storms", "clear weather"
    2. Equipment states: "PSO reports", "tripped", "reclosed"
    3. Network notifications: "DCC notified", "OMPA"
    4. Distance to fault: "DTF X.X miles"
    5. Cromwell Tap state: "Cromwell Tap 31 open/closed"

    Parameters:
    -----------
    unknown_events : pd.DataFrame
        Unknown events
    cause_col : str
        Name of cause column

    Returns:
    --------
    pd.DataFrame with boolean/categorical columns:
        - Weather_Mentioned: bool
        - Weather_In_Area: bool
        - Clear_Weather_Stated: bool
        - Storms_Mentioned: bool
        - PSO_Involvement: bool
        - OMPA_Involvement: bool
        - DCC_Notified: bool
        - Has_DTF_Info: bool
        - DTF_Miles: float (if mentioned)
        - Cromwell_Mentioned: bool
        - Cromwell_State: str
        - Suggested_Category: str (based on clue patterns)
    """
    unknown_with_clues = unknown_events.copy()

    # Extract text for analysis
    cause_text = unknown_with_clues[cause_col].fillna('')

    # Weather patterns
    unknown_with_clues['Weather_Mentioned'] = cause_text.str.contains(
        r'weather', case=False, regex=True
    )
    unknown_with_clues['Weather_In_Area'] = cause_text.str.contains(
        r'weather in.*area|weather.*area', case=False, regex=True
    )
    unknown_with_clues['Clear_Weather_Stated'] = cause_text.str.contains(
        r'clear weather|no.*weather', case=False, regex=True
    )
    unknown_with_clues['Storms_Mentioned'] = cause_text.str.contains(
        r'storm|severe weather|tornado|lightning', case=False, regex=True
    )

    # Entity involvement
    unknown_with_clues['PSO_Involvement'] = cause_text.str.contains(
        r'PSO', case=False, regex=False
    )
    unknown_with_clues['OMPA_Involvement'] = cause_text.str.contains(
        r'OMPA', case=False, regex=False
    )
    unknown_with_clues['DCC_Notified'] = cause_text.str.contains(
        r'DCC', case=False, regex=False
    )

    # Distance to fault
    def extract_dtf(text):
        """Extract DTF (Distance To Fault) miles from text."""
        if pd.isna(text):
            return None
        match = re.search(r'DTF[:\s]*(\d+\.?\d*)', str(text), re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    unknown_with_clues['DTF_Miles'] = cause_text.apply(extract_dtf)
    unknown_with_clues['Has_DTF_Info'] = unknown_with_clues['DTF_Miles'].notna()

    # Cromwell Tap
    unknown_with_clues['Cromwell_Mentioned'] = cause_text.str.contains(
        r'Cromwell', case=False, regex=True
    )

    def extract_cromwell_state(text):
        """Extract Cromwell Tap state."""
        if pd.isna(text):
            return 'not_mentioned'
        text_lower = str(text).lower()
        if 'cromwell' not in text_lower:
            return 'not_mentioned'
        if 'open' in text_lower:
            return 'open'
        elif 'closed' in text_lower:
            return 'closed'
        else:
            return 'mentioned_unclear'

    unknown_with_clues['Cromwell_State'] = cause_text.apply(extract_cromwell_state)

    # Suggest category based on clues
    def suggest_category(row):
        """Suggest reclassification based on clue patterns."""
        # Rule 1: Weather in area AND NOT clear weather → Weather, excluding lightning
        if row['Weather_In_Area'] and not row['Clear_Weather_Stated']:
            return 'Weather, excluding lightning'

        # Rule 2: Storms mentioned → Weather, excluding lightning
        if row['Storms_Mentioned']:
            return 'Weather, excluding lightning'

        # Rule 3: Cromwell mentioned with fault test/study → Maintenance/Testing
        cause_lower = str(row[cause_col]).lower()
        if row['Cromwell_Mentioned'] and ('fault test' in cause_lower or 'study' in cause_lower or 'studies' in cause_lower):
            return 'Maintenance/Testing'

        # Rule 4: Clear weather stated → Equipment/Other
        if row['Clear_Weather_Stated']:
            return 'Equipment/Other'

        # No strong suggestion
        return 'Insufficient clues'

    unknown_with_clues['Suggested_Category'] = unknown_with_clues.apply(suggest_category, axis=1)

    return unknown_with_clues


def propose_reclassification_rules(unknown_with_clues: pd.DataFrame) -> Dict:
    """
    Generate reclassification rules based on description patterns.

    Rules:
    1. IF "weather in area" AND NOT "clear weather" → "Weather, excluding lightning"
    2. IF "storms" OR "severe weather" → "Weather, excluding lightning"
    3. IF Cromwell mentioned AND "fault test" OR "study" → "Maintenance/Testing"
    4. IF "clear weather" → "Equipment/Other"

    Parameters:
    -----------
    unknown_with_clues : pd.DataFrame
        Unknown events with extracted clues

    Returns:
    --------
    Dict with:
        - total_unknown: int
        - reclassifiable_count: int
        - reclassifiable_pct: float
        - reclassification_rules: List[Dict]
        - suggested_distribution: Counter
    """
    total_unknown = len(unknown_with_clues)

    # Count by suggested category
    suggested_counts = unknown_with_clues['Suggested_Category'].value_counts()
    reclassifiable = suggested_counts.drop('Insufficient clues', errors='ignore').sum()

    # Define rules with counts
    rules = []

    # Rule 1: Weather in area
    rule1_count = ((unknown_with_clues['Weather_In_Area']) &
                   (~unknown_with_clues['Clear_Weather_Stated'])).sum()
    if rule1_count > 0:
        rules.append({
            'rule_name': 'Weather in area (not clear)',
            'pattern': 'Weather_In_Area AND NOT Clear_Weather_Stated',
            'suggested_category': 'Weather, excluding lightning',
            'confidence': 0.8,
            'applicable_events': int(rule1_count)
        })

    # Rule 2: Storms mentioned
    rule2_count = unknown_with_clues['Storms_Mentioned'].sum()
    if rule2_count > 0:
        rules.append({
            'rule_name': 'Storms/severe weather mentioned',
            'pattern': 'Storms_Mentioned',
            'suggested_category': 'Weather, excluding lightning',
            'confidence': 0.9,
            'applicable_events': int(rule2_count)
        })

    # Rule 3: Cromwell + fault test/study
    rule3_count = (unknown_with_clues['Suggested_Category'] == 'Maintenance/Testing').sum()
    if rule3_count > 0:
        rules.append({
            'rule_name': 'Cromwell Tap maintenance/testing',
            'pattern': 'Cromwell_Mentioned AND (fault test OR study)',
            'suggested_category': 'Maintenance/Testing',
            'confidence': 0.95,
            'applicable_events': int(rule3_count)
        })

    # Rule 4: Clear weather
    rule4_count = (unknown_with_clues['Suggested_Category'] == 'Equipment/Other').sum()
    if rule4_count > 0:
        rules.append({
            'rule_name': 'Clear weather stated',
            'pattern': 'Clear_Weather_Stated',
            'suggested_category': 'Equipment/Other',
            'confidence': 0.6,
            'applicable_events': int(rule4_count)
        })

    return {
        'total_unknown': int(total_unknown),
        'reclassifiable_count': int(reclassifiable),
        'reclassifiable_pct': (reclassifiable / total_unknown * 100) if total_unknown > 0 else 0,
        'reclassification_rules': rules,
        'suggested_distribution': Counter(suggested_counts.to_dict()),
        'summary': f"{reclassifiable} of {total_unknown} Unknown events ({reclassifiable/total_unknown*100:.1f}%) can be reclassified based on description clues"
    }


def get_unknown_analysis_summary(section150_events: pd.DataFrame,
                                  network_baselines: Dict) -> Dict:
    """
    Complete Unknown events analysis for Section 150.

    Combines all Unknown event analyses:
    - Filtering to Unknown events
    - Temporal clustering
    - Time-lagged correlation with known events
    - Operations field comparison
    - Description clue extraction
    - Reclassification rules

    Parameters:
    -----------
    section150_events : pd.DataFrame
        All Section 150 events
    network_baselines : Dict
        Network baseline statistics

    Returns:
    --------
    Dict with keys:
        - unknown_events: DataFrame
        - temporal_clustering: Dict
        - time_lagged: DataFrame
        - operations_comparison: Dict
        - unknown_with_clues: DataFrame
        - reclassification: Dict
    """
    cause_col = network_baselines['cause_col']
    datetime_col = network_baselines['datetime_col']

    # Filter to Unknown events
    unknown_events = filter_unknown_events(section150_events, cause_col)

    # Temporal clustering
    temporal_clustering = analyze_unknown_temporal_clustering(unknown_events, datetime_col)

    # Known events (for time-lagged correlation)
    known_events = section150_events[~section150_events.index.isin(unknown_events.index)].copy()

    # Parse Cause_Category for known events if not already done
    if 'Cause_Category' not in known_events.columns:
        def parse_category(text):
            if pd.isna(text):
                return 'Unknown'
            text = str(text).strip()
            if ' - ' in text:
                return text.split(' - ')[0].strip()
            return text

        known_events['Cause_Category'] = known_events[cause_col].apply(parse_category)

    # Time-lagged correlation
    time_lagged = find_time_lagged_correlations(
        unknown_events, known_events, datetime_col,
        lag_hours=cfg.UNKNOWN_TIME_LAG_HOURS if hasattr(cfg, 'UNKNOWN_TIME_LAG_HOURS') else 48
    )

    # Operations field comparison
    operations_comparison = analyze_unknown_operations_field(unknown_events, section150_events)

    # Extract description clues
    unknown_with_clues = extract_unknown_description_clues(unknown_events, cause_col)

    # Reclassification rules
    reclassification = propose_reclassification_rules(unknown_with_clues)

    return {
        'unknown_events': unknown_events,
        'total_unknown': len(unknown_events),
        'temporal_clustering': temporal_clustering,
        'time_lagged': time_lagged,
        'operations_comparison': operations_comparison,
        'unknown_with_clues': unknown_with_clues,
        'reclassification': reclassification
    }
