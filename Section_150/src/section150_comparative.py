"""
Comparative analysis for Section 150.
Find similar sections and analyze why they have fewer events.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_section150 as cfg

from section150_characteristics import (
    find_similar_sections,
    identify_voltage_column,
    identify_type_column,
    extract_pmu_details
)


def get_section_event_profile(disturbance_df: pd.DataFrame,
                              section_id: int,
                              cause_col: str,
                              datetime_col: str) -> Dict:
    """
    Get event profile for a single section.
    
    Returns:
    --------
    Dict: Event profile including counts, causes, timing
    """
    section_events = disturbance_df[disturbance_df['SectionID'] == section_id]
    
    if section_events.empty:
        return {'event_count': 0, 'section_id': section_id}
    
    # Cause distribution
    causes = section_events[cause_col].value_counts()
    top_cause = causes.index[0] if len(causes) > 0 else None
    
    # Temporal info
    dates = section_events[datetime_col]
    first_event = dates.min()
    last_event = dates.max()
    
    # Inter-arrival times
    sorted_dates = dates.sort_values()
    iat = sorted_dates.diff().dt.total_seconds() / 3600  # hours
    mean_iat = iat.mean() if len(iat) > 1 else None
    
    return {
        'section_id': section_id,
        'event_count': len(section_events),
        'top_cause': top_cause,
        'top_cause_pct': (causes.iloc[0] / len(section_events) * 100) if len(causes) > 0 else 0,
        'cause_diversity': len(causes),
        'first_event': first_event,
        'last_event': last_event,
        'mean_iat_hours': mean_iat,
        'causes': causes
    }


def compare_similar_sections_detailed(section150_events: pd.DataFrame,
                                      disturbance_df: pd.DataFrame,
                                      pmu_df: pd.DataFrame,
                                      cause_col: str,
                                      datetime_col: str,
                                      n_similar: int = 10) -> pd.DataFrame:
    """
    Detailed comparison of Section 150 with similar sections.
    
    Returns:
    --------
    pd.DataFrame: Comparison table with detailed metrics
    """
    # Find similar sections
    similar = find_similar_sections(pmu_df, disturbance_df, cfg.TARGET_SECTION_ID, n_similar)
    
    if similar.empty:
        return pd.DataFrame()
    
    # Get Section 150 profile
    sec150_profile = get_section_event_profile(
        disturbance_df, cfg.TARGET_SECTION_ID, cause_col, datetime_col
    )
    
    # Get profiles for similar sections
    profiles = []
    for section_id in similar['SectionID'].values:
        profile = get_section_event_profile(disturbance_df, section_id, cause_col, datetime_col)
        profile['Similarity'] = similar[similar['SectionID'] == section_id]['Similarity'].values[0]
        profiles.append(profile)
    
    comparison_df = pd.DataFrame(profiles)
    
    # Add comparisons to Section 150
    comparison_df['Event_Ratio_vs_150'] = sec150_profile['event_count'] / comparison_df['event_count'].replace(0, 1)
    comparison_df['Same_Top_Cause'] = comparison_df['top_cause'] == sec150_profile['top_cause']
    
    # Merge with PMU characteristics
    pmu_cols = ['SectionID']
    voltage_col = identify_voltage_column(pmu_df)
    type_col = identify_type_column(pmu_df)
    
    if voltage_col:
        pmu_cols.append(voltage_col)
    if type_col:
        pmu_cols.append(type_col)
    if 'InService' in pmu_df.columns:
        pmu_cols.append('InService')
    
    comparison_df = comparison_df.merge(
        pmu_df[pmu_cols],
        left_on='section_id',
        right_on='SectionID',
        how='left'
    )
    
    return comparison_df


def analyze_what_differs(section150_profile: Dict,
                         comparison_df: pd.DataFrame,
                         pmu_df: pd.DataFrame) -> Dict:
    """
    Analyze what makes Section 150 different from similar sections.
    
    Returns:
    --------
    Dict: Analysis of differences
    """
    differences = {
        'event_count_comparison': {},
        'cause_comparison': {},
        'timing_comparison': {},
        'key_differences': []
    }
    
    if comparison_df.empty:
        return differences
    
    # Event count comparison
    similar_mean_events = comparison_df['event_count'].mean()
    similar_max_events = comparison_df['event_count'].max()
    sec150_events = section150_profile['event_count']
    
    differences['event_count_comparison'] = {
        'section150': sec150_events,
        'similar_mean': similar_mean_events,
        'similar_max': similar_max_events,
        'ratio_to_mean': sec150_events / similar_mean_events if similar_mean_events > 0 else float('inf'),
        'ratio_to_max': sec150_events / similar_max_events if similar_max_events > 0 else float('inf')
    }
    
    # Add key difference
    if sec150_events > similar_mean_events * 2:
        differences['key_differences'].append(
            f"Section 150 has {sec150_events / similar_mean_events:.1f}x more events than similar sections average ({similar_mean_events:.0f})"
        )
    
    # Cause comparison
    sec150_top_cause = section150_profile['top_cause']
    similar_top_causes = comparison_df['top_cause'].value_counts()
    most_common_similar = similar_top_causes.index[0] if len(similar_top_causes) > 0 else None
    
    differences['cause_comparison'] = {
        'section150_top_cause': sec150_top_cause,
        'similar_sections_common_cause': most_common_similar,
        'same_top_cause_pct': (comparison_df['Same_Top_Cause'].sum() / len(comparison_df) * 100) if len(comparison_df) > 0 else 0
    }
    
    if sec150_top_cause != most_common_similar:
        differences['key_differences'].append(
            f"Section 150's top cause is '{sec150_top_cause}' while similar sections typically have '{most_common_similar}'"
        )
    
    # Inter-arrival time comparison
    sec150_iat = section150_profile.get('mean_iat_hours')
    similar_iat_mean = comparison_df['mean_iat_hours'].mean()
    
    if sec150_iat and similar_iat_mean:
        differences['timing_comparison'] = {
            'section150_mean_iat_hours': sec150_iat,
            'similar_mean_iat_hours': similar_iat_mean,
            'iat_ratio': similar_iat_mean / sec150_iat if sec150_iat > 0 else None
        }
        
        if sec150_iat < similar_iat_mean / 2:
            differences['key_differences'].append(
                f"Section 150 has events {similar_iat_mean/sec150_iat:.1f}x more frequently than similar sections"
            )
    
    return differences


def generate_learnings(differences: Dict, comparison_df: pd.DataFrame) -> List[str]:
    """
    Generate actionable learnings from the comparison.
    
    Returns:
    --------
    List[str]: Key learnings and insights
    """
    learnings = []
    
    if comparison_df.empty:
        return ["Insufficient data for comparative analysis"]
    
    # Find best-performing similar section
    best_section = comparison_df.loc[comparison_df['event_count'].idxmin()]
    
    learnings.append(
        f"Best-performing similar section ({int(best_section['section_id'])}) has only "
        f"{int(best_section['event_count'])} events vs Section 150's {differences['event_count_comparison']['section150']}"
    )
    
    # Cause-based learning
    cause_comp = differences.get('cause_comparison', {})
    if cause_comp.get('same_top_cause_pct', 100) < 50:
        learnings.append(
            f"Section 150's dominant cause pattern differs from most similar sections - investigate cause-specific interventions"
        )
    
    # Timing-based learning
    timing_comp = differences.get('timing_comparison', {})
    if timing_comp.get('iat_ratio') and timing_comp['iat_ratio'] > 2:
        learnings.append(
            f"Events at Section 150 occur {timing_comp['iat_ratio']:.1f}x more frequently - consider increased monitoring"
        )
    
    # General learnings from differences
    for diff in differences.get('key_differences', []):
        learnings.append(f"KEY: {diff}")
    
    return learnings


def get_comparative_summary(section150_events: pd.DataFrame,
                            disturbance_df: pd.DataFrame,
                            pmu_df: pd.DataFrame,
                            network_baselines: Dict) -> Dict:
    """
    Complete comparative analysis for Section 150.
    
    Returns:
    --------
    Dict: All comparative analyses
    """
    cause_col = network_baselines['cause_col']
    datetime_col = network_baselines['datetime_col']
    
    # Get Section 150 profile
    sec150_profile = get_section_event_profile(
        disturbance_df, cfg.TARGET_SECTION_ID, cause_col, datetime_col
    )
    
    # Detailed comparison with similar sections
    comparison_df = compare_similar_sections_detailed(
        section150_events, disturbance_df, pmu_df, cause_col, datetime_col
    )
    
    # Analyze differences
    differences = analyze_what_differs(sec150_profile, comparison_df, pmu_df)
    
    # Generate learnings
    learnings = generate_learnings(differences, comparison_df)
    
    return {
        'section150_profile': sec150_profile,
        'similar_sections': comparison_df,
        'differences': differences,
        'learnings': learnings,
        'n_similar_analyzed': len(comparison_df)
    }
