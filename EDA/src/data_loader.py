"""
Data loading utilities for PMU disturbance analysis.
Handles Excel data import, date parsing, and data merging.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings

def load_pmu_data(file_path: str, sheet_name: str = 'PMUs') -> pd.DataFrame:
    """
    Load PMU installation data from Excel.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    sheet_name : str
        Name of the sheet containing PMU data (default: 'PMUs')

    Returns:
    --------
    pd.DataFrame
        PMU data with parsed dates
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Parse date columns
    date_columns = ['InService', 'OutService']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def load_disturbance_data(file_path: str, sheet_name: str = 'Disturbances') -> pd.DataFrame:
    """
    Load disturbance event data from Excel.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    sheet_name : str
        Name of the sheet containing disturbance data (default: 'Disturbances')

    Returns:
    --------
    pd.DataFrame
        Disturbance data with parsed timestamps
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Identify and parse datetime columns
    # Common datetime column names
    datetime_cols = [col for col in df.columns if any(
        keyword in col.lower() for keyword in ['date', 'time', 'timestamp']
    )]

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def merge_pmu_disturbance(pmu_df: pd.DataFrame,
                          disturbance_df: pd.DataFrame,
                          join_key: str = 'SectionID') -> pd.DataFrame:
    """
    Merge PMU and disturbance data on SectionID.

    Parameters:
    -----------
    pmu_df : pd.DataFrame
        PMU installation data
    disturbance_df : pd.DataFrame
        Disturbance event data
    join_key : str
        Column name to join on (default: 'SectionID')

    Returns:
    --------
    pd.DataFrame
        Merged data with PMU and disturbance information
    """
    if join_key not in pmu_df.columns:
        raise ValueError(f"Join key '{join_key}' not found in PMU data")
    if join_key not in disturbance_df.columns:
        raise ValueError(f"Join key '{join_key}' not found in disturbance data")

    # Perform left join to keep all disturbances
    merged = disturbance_df.merge(pmu_df, on=join_key, how='left', suffixes=('_dist', '_pmu'))

    # Check for unmatched records
    unmatched = merged[join_key].isna().sum()
    if unmatched > 0:
        warnings.warn(f"{unmatched} disturbance records could not be matched to PMU data")

    return merged


def load_all_data(file_path: str,
                  pmu_sheet: str = 'PMUs',
                  disturbance_sheet: str = 'Disturbances') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and return all datasets: PMU data, disturbance data, and merged data.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    pmu_sheet : str
        Name of the PMU sheet
    disturbance_sheet : str
        Name of the disturbance sheet

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (pmu_df, disturbance_df, merged_df)
    """
    print("Loading PMU data...")
    pmu_df = load_pmu_data(file_path, pmu_sheet)
    print(f"Loaded {len(pmu_df)} PMU records")

    print("Loading disturbance data...")
    disturbance_df = load_disturbance_data(file_path, disturbance_sheet)
    print(f"Loaded {len(disturbance_df)} disturbance records")

    print("Merging datasets...")
    merged_df = merge_pmu_disturbance(pmu_df, disturbance_df)
    print(f"Merged dataset contains {len(merged_df)} records")

    return pmu_df, disturbance_df, merged_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get a comprehensive summary of the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Summary statistics including shape, dtypes, missing values, etc.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

    return summary
