"""
Data loading utilities specific to Section 150 analysis.
Wraps EDA data_loader and adds Section 150-specific filtering and caching.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'EDA'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'EDA', 'src'))

from data_loader import load_all_data, load_pmu_data, load_disturbance_data
import config_section150 as cfg


def load_section150_data(use_cache: bool = True) -> tuple:
    """
    Load Section 150 events and full network data.
    
    Returns:
    --------
    tuple: (section150_events, all_disturbances, pmu_df, merged_df)
    """
    cache_csv = cfg.CACHED_SECTION150_DATA.replace('.parquet', '.csv')
    
    # Check cache
    if use_cache and os.path.exists(cache_csv):
        print("Loading cached Section 150 data...")
        section150_events = pd.read_csv(cache_csv, parse_dates=True)
        # Still need to load full data for comparisons
        pmu_df, disturbance_df, merged_df = load_all_data(cfg.EXCEL_FILE, cfg.PMU_SHEET, cfg.DISTURBANCE_SHEET)
        return section150_events, disturbance_df, pmu_df, merged_df
    
    # Load all data
    print("Loading data from Excel...")
    pmu_df, disturbance_df, merged_df = load_all_data(cfg.EXCEL_FILE, cfg.PMU_SHEET, cfg.DISTURBANCE_SHEET)
    
    # Filter for Section 150
    section150_events = disturbance_df[disturbance_df['SectionID'] == cfg.TARGET_SECTION_ID].copy()
    print(f"Section 150: {len(section150_events)} disturbance events")
    
    # Cache Section 150 data
    Path(cfg.DATA_DIR).mkdir(parents=True, exist_ok=True)
    section150_events.to_csv(cache_csv, index=False)
    
    return section150_events, disturbance_df, pmu_df, merged_df


def compute_network_baselines(disturbance_df: pd.DataFrame, pmu_df: pd.DataFrame) -> dict:
    """
    Compute network-wide statistics for comparison with Section 150.
    
    Returns:
    --------
    dict: Network baseline statistics
    """
    total_events = len(disturbance_df)
    total_sections = disturbance_df['SectionID'].nunique()
    
    # Events per section
    events_per_section = disturbance_df.groupby('SectionID').size()
    
    # Identify cause column
    cause_col = None
    for col in disturbance_df.columns:
        if 'cause' in col.lower():
            cause_col = col
            break
    
    # Cause distribution (network-wide)
    cause_distribution = None
    if cause_col:
        cause_distribution = disturbance_df[cause_col].value_counts(normalize=True)
    
    # Identify datetime column
    datetime_col = None
    for col in disturbance_df.columns:
        if disturbance_df[col].dtype == 'datetime64[ns]':
            datetime_col = col
            break
    
    # Time range
    time_range = None
    if datetime_col:
        time_range = (disturbance_df[datetime_col].min(), disturbance_df[datetime_col].max())
    
    baselines = {
        'total_events': total_events,
        'total_sections': total_sections,
        'mean_events_per_section': events_per_section.mean(),
        'median_events_per_section': events_per_section.median(),
        'std_events_per_section': events_per_section.std(),
        'max_events_per_section': events_per_section.max(),
        'min_events_per_section': events_per_section.min(),
        'events_per_section': events_per_section,
        'cause_distribution': cause_distribution,
        'cause_col': cause_col,
        'datetime_col': datetime_col,
        'time_range': time_range
    }
    
    return baselines


def get_section150_pmu_info(pmu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract PMU metadata for Section 150.
    
    Returns:
    --------
    pd.DataFrame: PMU information for Section 150
    """
    section150_pmu = pmu_df[pmu_df['SectionID'] == cfg.TARGET_SECTION_ID].copy()
    
    # Calculate age if InService date exists
    if 'InService' in section150_pmu.columns:
        section150_pmu['Age_Years'] = (pd.Timestamp.now() - section150_pmu['InService']).dt.days / 365.25
    
    return section150_pmu


def get_similar_sections(pmu_df: pd.DataFrame, disturbance_df: pd.DataFrame, 
                         n_similar: int = 10) -> pd.DataFrame:
    """
    Find sections similar to Section 150 based on PMU characteristics.
    
    Returns:
    --------
    pd.DataFrame: Similar sections with their characteristics
    """
    # Get Section 150 PMU info
    sec150_pmu = get_section150_pmu_info(pmu_df)
    
    if sec150_pmu.empty:
        print("Warning: No PMU data found for Section 150")
        return pd.DataFrame()
    
    # Extract key features for similarity
    features_to_compare = []
    
    # Voltage level (most important for similarity)
    voltage_col = None
    for col in pmu_df.columns:
        if 'volt' in col.lower() or 'kv' in col.lower():
            voltage_col = col
            break
    
    # PMU Type
    type_col = None
    for col in pmu_df.columns:
        if 'type' in col.lower():
            type_col = col
            break
    
    # Get Section 150's characteristics
    sec150_voltage = sec150_pmu[voltage_col].values[0] if voltage_col and not sec150_pmu[voltage_col].isna().all() else None
    sec150_type = sec150_pmu[type_col].values[0] if type_col and not sec150_pmu[type_col].isna().all() else None
    
    # Filter similar sections
    similar_mask = pd.Series([True] * len(pmu_df), index=pmu_df.index)
    
    if sec150_voltage is not None and voltage_col:
        similar_mask &= pmu_df[voltage_col] == sec150_voltage
    
    if sec150_type is not None and type_col:
        similar_mask &= pmu_df[type_col] == sec150_type
    
    # Exclude Section 150 itself
    similar_mask &= pmu_df['SectionID'] != cfg.TARGET_SECTION_ID
    
    similar_pmus = pmu_df[similar_mask].copy()
    
    # Add event counts
    events_per_section = disturbance_df.groupby('SectionID').size()
    similar_pmus['Event_Count'] = similar_pmus['SectionID'].map(events_per_section).fillna(0).astype(int)
    
    # Sort by event count (to find sections with fewer events for comparison)
    similar_pmus = similar_pmus.sort_values('Event_Count', ascending=True)
    
    return similar_pmus.head(n_similar)


def get_all_sections_summary(disturbance_df: pd.DataFrame, pmu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all sections for comparison.
    
    Returns:
    --------
    pd.DataFrame: Summary of all sections
    """
    # Event counts per section
    events_per_section = disturbance_df.groupby('SectionID').size().reset_index(name='Event_Count')
    
    # Merge with PMU info
    summary = events_per_section.merge(pmu_df, on='SectionID', how='left')
    
    # Calculate age if InService date exists
    if 'InService' in summary.columns:
        summary['Age_Years'] = (pd.Timestamp.now() - summary['InService']).dt.days / 365.25
    
    # Mark Section 150
    summary['Is_Section_150'] = summary['SectionID'] == cfg.TARGET_SECTION_ID
    
    return summary.sort_values('Event_Count', ascending=False)
