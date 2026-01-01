"""
PMU characteristics analysis for Section 150.
Extracts details, compares to similar sections, identifies unique features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_section150 as cfg


def extract_pmu_details(pmu_df: pd.DataFrame, section_id: int = 150) -> Dict:
    """
    Extract all PMU details for a given section.
    
    Returns:
    --------
    Dict: PMU characteristics
    """
    section_pmu = pmu_df[pmu_df['SectionID'] == section_id]
    
    if section_pmu.empty:
        return {'error': f'No PMU data found for Section {section_id}'}
    
    details = {}
    for col in section_pmu.columns:
        val = section_pmu[col].values[0]
        # Handle different types
        if pd.isna(val):
            details[col] = None
        elif isinstance(val, (np.datetime64, pd.Timestamp)):
            details[col] = pd.Timestamp(val)
        else:
            details[col] = val
    
    # Calculate age
    if 'InService' in details and details['InService'] is not None:
        details['Age_Years'] = (pd.Timestamp.now() - details['InService']).days / 365.25
        details['Age_Days'] = (pd.Timestamp.now() - details['InService']).days
    
    return details


def identify_voltage_column(pmu_df: pd.DataFrame) -> str:
    """Identify the voltage column in PMU data."""
    for col in pmu_df.columns:
        if 'volt' in col.lower() or 'kv' in col.lower():
            return col
    return None


def identify_type_column(pmu_df: pd.DataFrame) -> str:
    """Identify the PMU type column."""
    for col in pmu_df.columns:
        if 'type' in col.lower():
            return col
    return None


def identify_location_columns(pmu_df: pd.DataFrame) -> Tuple[str, str]:
    """Identify latitude and longitude columns."""
    lat_col, lon_col = None, None
    for col in pmu_df.columns:
        col_lower = col.lower()
        if 'lat' in col_lower:
            lat_col = col
        elif 'lon' in col_lower or 'lng' in col_lower:
            lon_col = col
    return lat_col, lon_col


def compute_section_features(pmu_df: pd.DataFrame, disturbance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature matrix for all sections for similarity comparison.
    
    Returns:
    --------
    pd.DataFrame: Feature matrix with SectionID as index
    """
    features = pmu_df.copy()
    
    # Add event counts
    event_counts = disturbance_df.groupby('SectionID').size()
    features['Event_Count'] = features['SectionID'].map(event_counts).fillna(0)
    
    # Calculate age
    if 'InService' in features.columns:
        features['Age_Years'] = (pd.Timestamp.now() - features['InService']).dt.days / 365.25
    
    # Identify key columns
    voltage_col = identify_voltage_column(features)
    type_col = identify_type_column(features)
    lat_col, lon_col = identify_location_columns(features)
    
    # Create numeric feature matrix
    numeric_features = ['Event_Count']
    
    if 'Age_Years' in features.columns:
        numeric_features.append('Age_Years')
    
    if voltage_col and features[voltage_col].dtype in ['int64', 'float64']:
        numeric_features.append(voltage_col)
    
    if lat_col and lon_col:
        numeric_features.extend([lat_col, lon_col])
    
    # One-hot encode categorical features
    if type_col:
        type_dummies = pd.get_dummies(features[type_col], prefix='Type')
        features = pd.concat([features, type_dummies], axis=1)
        numeric_features.extend(type_dummies.columns.tolist())
    
    if voltage_col and features[voltage_col].dtype == 'object':
        volt_dummies = pd.get_dummies(features[voltage_col], prefix='Voltage')
        features = pd.concat([features, volt_dummies], axis=1)
        numeric_features.extend(volt_dummies.columns.tolist())
    
    features = features.set_index('SectionID')
    
    return features, numeric_features


def find_similar_sections(pmu_df: pd.DataFrame, 
                          disturbance_df: pd.DataFrame,
                          target_section: int = 150,
                          n_similar: int = 10,
                          exclude_event_count: bool = True) -> pd.DataFrame:
    """
    Find sections most similar to target based on PMU characteristics.
    
    Parameters:
    -----------
    exclude_event_count : bool
        If True, don't use event count for similarity (avoids leakage)
    
    Returns:
    --------
    pd.DataFrame: Similar sections ranked by similarity
    """
    features, numeric_cols = compute_section_features(pmu_df, disturbance_df)
    
    if target_section not in features.index:
        return pd.DataFrame()
    
    # Exclude event count if requested
    if exclude_event_count and 'Event_Count' in numeric_cols:
        similarity_cols = [c for c in numeric_cols if c != 'Event_Count']
    else:
        similarity_cols = numeric_cols
    
    # Get numeric data
    X = features[similarity_cols].fillna(0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    # Get target vector
    target_vec = X_scaled_df.loc[target_section].values.reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(target_vec, X_scaled_df.values)[0]
    
    # Create results dataframe
    results = pd.DataFrame({
        'SectionID': features.index,
        'Similarity': similarities
    })
    
    # Merge with feature data
    results = results.merge(features.reset_index()[['SectionID', 'Event_Count'] + 
                            [c for c in features.columns if c in pmu_df.columns]], 
                            on='SectionID', how='left')
    
    # Exclude target section and sort
    results = results[results['SectionID'] != target_section]
    results = results.sort_values('Similarity', ascending=False)
    
    return results.head(n_similar)


def compare_section_characteristics(section150_details: Dict,
                                    similar_sections: pd.DataFrame,
                                    pmu_df: pd.DataFrame) -> Dict:
    """
    Compare Section 150 characteristics to similar sections.
    
    Returns:
    --------
    Dict: Comparison results
    """
    comparison = {
        'section150': section150_details,
        'similar_sections_summary': {}
    }
    
    # Calculate statistics for similar sections
    if not similar_sections.empty:
        comparison['similar_sections_summary'] = {
            'mean_event_count': similar_sections['Event_Count'].mean(),
            'median_event_count': similar_sections['Event_Count'].median(),
            'max_event_count': similar_sections['Event_Count'].max(),
            'min_event_count': similar_sections['Event_Count'].min()
        }
        
        # Event count ratio
        sec150_events = section150_details.get('Event_Count', 301)
        comparison['event_ratio_vs_similar'] = sec150_events / comparison['similar_sections_summary']['mean_event_count']
    
    return comparison


def identify_unique_features(section150_details: Dict,
                             pmu_df: pd.DataFrame,
                             disturbance_df: pd.DataFrame) -> List[str]:
    """
    Identify what makes Section 150 unique compared to other sections.
    
    Returns:
    --------
    List[str]: List of unique/notable characteristics
    """
    unique_features = []
    
    # Get network statistics
    event_counts = disturbance_df.groupby('SectionID').size()
    
    # Event count ranking
    sec150_events = event_counts.get(cfg.TARGET_SECTION_ID, 0)
    rank = (event_counts > sec150_events).sum() + 1
    unique_features.append(f"Rank #{rank} in event frequency with {sec150_events} events (network max: {event_counts.max()})")
    
    # Age analysis
    if 'InService' in pmu_df.columns:
        pmu_ages = (pd.Timestamp.now() - pmu_df['InService']).dt.days / 365.25
        sec150_age = section150_details.get('Age_Years')
        if sec150_age:
            age_percentile = (pmu_ages < sec150_age).sum() / len(pmu_ages) * 100
            if age_percentile > 75:
                unique_features.append(f"Older than {age_percentile:.0f}% of sections ({sec150_age:.1f} years)")
            elif age_percentile < 25:
                unique_features.append(f"Newer than {100-age_percentile:.0f}% of sections ({sec150_age:.1f} years)")
    
    # Voltage level
    voltage_col = identify_voltage_column(pmu_df)
    if voltage_col and voltage_col in section150_details:
        sec150_voltage = section150_details[voltage_col]
        voltage_counts = pmu_df[voltage_col].value_counts(normalize=True) * 100
        voltage_pct = voltage_counts.get(sec150_voltage, 0)
        unique_features.append(f"Voltage level {sec150_voltage}: {voltage_pct:.1f}% of network")
    
    # PMU Type
    type_col = identify_type_column(pmu_df)
    if type_col and type_col in section150_details:
        sec150_type = section150_details[type_col]
        type_counts = pmu_df[type_col].value_counts(normalize=True) * 100
        type_pct = type_counts.get(sec150_type, 0)
        unique_features.append(f"PMU Type '{sec150_type}': {type_pct:.1f}% of network")
    
    return unique_features


def get_characteristics_summary(pmu_df: pd.DataFrame,
                                disturbance_df: pd.DataFrame) -> Dict:
    """
    Complete Section 150 characteristics analysis.
    
    Returns:
    --------
    Dict: All characteristics analyses
    """
    # Extract Section 150 details
    sec150_details = extract_pmu_details(pmu_df, cfg.TARGET_SECTION_ID)
    
    # Add event count
    event_counts = disturbance_df.groupby('SectionID').size()
    sec150_details['Event_Count'] = event_counts.get(cfg.TARGET_SECTION_ID, 0)
    
    # Find similar sections
    similar = find_similar_sections(pmu_df, disturbance_df, cfg.TARGET_SECTION_ID, cfg.N_SIMILAR_SECTIONS)
    
    # Compare characteristics
    comparison = compare_section_characteristics(sec150_details, similar, pmu_df)
    
    # Identify unique features
    unique = identify_unique_features(sec150_details, pmu_df, disturbance_df)
    
    return {
        'section150_details': sec150_details,
        'similar_sections': similar,
        'comparison': comparison,
        'unique_features': unique,
        'voltage_col': identify_voltage_column(pmu_df),
        'type_col': identify_type_column(pmu_df)
    }
