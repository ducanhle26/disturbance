"""
Data Loading Module for PMU Reliability Framework.

Handles loading, validation, and basic statistics for PMU disturbance data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Any


def load_pmu_disturbance_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PMU disturbance data from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to PMU_disturbance.xlsx file
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (pmu_df, dist_df) - PMU installations and disturbance events
        
    Raises
    ------
    FileNotFoundError
        If the Excel file does not exist
    ValueError
        If required sheets or columns are missing
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    pmu_df = pd.read_excel(filepath, sheet_name='PMUs')
    dist_df = pd.read_excel(filepath, sheet_name='Disturbances')
    
    if 'SectionID' not in pmu_df.columns:
        raise ValueError("PMU data missing required column: SectionID")
    if 'SectionID' not in dist_df.columns:
        raise ValueError("Disturbance data missing required column: SectionID")
    
    for col in ['InService', 'OutService']:
        if col in pmu_df.columns:
            pmu_df[col] = pd.to_datetime(pmu_df[col], errors='coerce')
    
    datetime_cols = [col for col in dist_df.columns 
                     if any(k in col.lower() for k in ['date', 'time', 'timestamp'])]
    for col in datetime_cols:
        dist_df[col] = pd.to_datetime(dist_df[col], errors='coerce')
    
    if 'InService' in pmu_df.columns:
        pmu_df['Age_Days'] = (pd.Timestamp.now() - pmu_df['InService']).dt.days
        pmu_df['Age_Years'] = pmu_df['Age_Days'] / 365.25
    
    return pmu_df, dist_df


def get_section_events(dist_df: pd.DataFrame, section_id: int) -> pd.DataFrame:
    """
    Extract disturbance events for a specific section.
    
    Parameters
    ----------
    dist_df : pd.DataFrame
        Full disturbance dataframe
    section_id : int
        Section ID to filter for
        
    Returns
    -------
    pd.DataFrame
        Filtered events for the specified section
    """
    return dist_df[dist_df['SectionID'] == section_id].copy()


def calculate_event_statistics(events_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for a set of events.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Disturbance events dataframe
        
    Returns
    -------
    Dict[str, Any]
        Statistics including count, mtbf_days, first_event, last_event
    """
    count = len(events_df)
    
    if count == 0:
        return {
            'count': 0,
            'mtbf_days': None,
            'first_event': None,
            'last_event': None,
            'span_days': None
        }
    
    datetime_col = _find_datetime_column(events_df)
    
    if datetime_col is None or count < 2:
        return {
            'count': count,
            'mtbf_days': None,
            'first_event': events_df[datetime_col].min() if datetime_col else None,
            'last_event': events_df[datetime_col].max() if datetime_col else None,
            'span_days': None
        }
    
    sorted_events = events_df.sort_values(datetime_col)
    first_event = sorted_events[datetime_col].min()
    last_event = sorted_events[datetime_col].max()
    span_days = (last_event - first_event).days
    
    inter_arrival = sorted_events[datetime_col].diff().dropna()
    mtbf_days = inter_arrival.dt.total_seconds().mean() / 86400  # seconds to days
    
    return {
        'count': count,
        'mtbf_days': mtbf_days,
        'first_event': first_event,
        'last_event': last_event,
        'span_days': span_days
    }


def get_network_statistics(dist_df: pd.DataFrame, pmu_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate network-wide statistics for comparison.
    
    Parameters
    ----------
    dist_df : pd.DataFrame
        Full disturbance dataframe
    pmu_df : pd.DataFrame
        PMU installations dataframe
        
    Returns
    -------
    Dict[str, Any]
        Network-wide statistics
    """
    total_events = len(dist_df)
    total_sections = dist_df['SectionID'].nunique()
    total_pmus = len(pmu_df)
    
    events_per_section = dist_df.groupby('SectionID').size()
    
    cause_col = _find_cause_column(dist_df)
    cause_distribution = dist_df[cause_col].value_counts() if cause_col else None
    
    datetime_col = _find_datetime_column(dist_df)
    time_range = None
    if datetime_col:
        time_range = (dist_df[datetime_col].min(), dist_df[datetime_col].max())
    
    return {
        'total_events': total_events,
        'total_sections': total_sections,
        'total_pmus': total_pmus,
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


def _find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Find the primary datetime column in a dataframe."""
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            return col
    return None


def _find_cause_column(df: pd.DataFrame) -> Optional[str]:
    """Find the cause column in a dataframe."""
    for col in df.columns:
        if 'cause' in col.lower():
            return col
    return None
