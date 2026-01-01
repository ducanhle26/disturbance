"""
Operational context analysis for Section 150.
Extracts operational intelligence from event text descriptions.
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


def parse_cause_field(cause_text: str) -> Dict[str, str]:
    """
    Parse Cause field into category and description.

    Format: "Category - Description" or "Category"

    Parameters:
    -----------
    cause_text : str
        Full cause text from event record

    Returns:
    --------
    Dict with keys:
        - category: str (e.g., "Unknown", "Weather, excluding lightning")
        - description: str (detailed context after dash, or empty string)
        - has_description: bool

    Examples:
    ---------
    >>> parse_cause_field("Unknown - Fixico 9 opened and re-closed")
    {'category': 'Unknown', 'description': 'Fixico 9 opened and re-closed', 'has_description': True}

    >>> parse_cause_field("Lightning")
    {'category': 'Lightning', 'description': '', 'has_description': False}
    """
    if pd.isna(cause_text) or cause_text == '':
        return {'category': 'Unknown', 'description': '', 'has_description': False}

    cause_text = str(cause_text).strip()

    # Split on first " - " delimiter
    if ' - ' in cause_text:
        parts = cause_text.split(' - ', 1)
        category = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ''
        has_description = bool(description)
    else:
        category = cause_text
        description = ''
        has_description = False

    return {
        'category': category,
        'description': description,
        'has_description': has_description
    }


def extract_cromwell_tap_state(section150_events: pd.DataFrame,
                                cause_col: str) -> pd.DataFrame:
    """
    Text-mine Cromwell Tap 31 operational state from event descriptions.

    Uses regex patterns to detect:
    - "Cromwell Tap 31 open" → "open"
    - "Cromwell Tap 31 closed" → "closed"
    - "Cromwell 31 open" → "open"
    - No mention → "not_mentioned"

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events dataframe
    cause_col : str
        Name of the cause column

    Returns:
    --------
    pd.DataFrame with new columns:
        - Cause_Category: str (parsed category)
        - Cause_Description: str (parsed description)
        - Cromwell_Tap_State: str ["open", "closed", "not_mentioned"]
        - Cromwell_Mentioned: bool
    """
    events = section150_events.copy()

    # Parse cause field
    parsed_causes = events[cause_col].apply(parse_cause_field)
    events['Cause_Category'] = parsed_causes.apply(lambda x: x['category'])
    events['Cause_Description'] = parsed_causes.apply(lambda x: x['description'])

    # Define regex patterns for Cromwell Tap detection
    patterns_open = [
        r'Cromwell Tap 31 open',
        r'Cromwell 31 open',
        r'Cromwell.*31.*open',
        r'Tap 31.*open'
    ]

    patterns_closed = [
        r'Cromwell Tap 31 closed',
        r'Cromwell 31 closed',
        r'Cromwell.*31.*closed',
        r'Tap 31.*closed'
    ]

    def detect_cromwell_state(cause_text):
        """Detect Cromwell Tap 31 state from text."""
        if pd.isna(cause_text):
            return 'not_mentioned'

        cause_text_lower = str(cause_text).lower()

        # Check for "open" patterns
        for pattern in patterns_open:
            if re.search(pattern, cause_text_lower):
                return 'open'

        # Check for "closed" patterns
        for pattern in patterns_closed:
            if re.search(pattern, cause_text_lower):
                return 'closed'

        # Check if Cromwell is mentioned at all (but state unclear)
        if 'cromwell' in cause_text_lower:
            # Try to infer from context
            if 'open' in cause_text_lower:
                return 'open'
            elif 'closed' in cause_text_lower:
                return 'closed'
            else:
                return 'mentioned_unclear'

        return 'not_mentioned'

    # Apply state detection
    events['Cromwell_Tap_State'] = events[cause_col].apply(detect_cromwell_state)
    events['Cromwell_Mentioned'] = events['Cromwell_Tap_State'] != 'not_mentioned'

    return events


def analyze_cromwell_tap_correlation(events_with_state: pd.DataFrame,
                                      datetime_col: str) -> Dict:
    """
    Analyze correlation between Cromwell Tap state and event frequency.

    Statistical tests:
    - Event rate comparison: Events/day when open vs closed vs not mentioned
    - Chi-square: Is Cromwell Tap state distribution random?
    - Time-lagged: Do state changes precede event bursts?

    Parameters:
    -----------
    events_with_state : pd.DataFrame
        Events with Cromwell_Tap_State column
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    Dict with:
        - state_distribution: Counter
        - mentions: int (total events mentioning Cromwell)
        - events_when_open: int
        - events_when_closed: int
        - events_when_unclear: int
        - events_not_mentioned: int
        - pct_when_open: float
        - pct_when_closed: float
        - rate_analysis: Dict (event rates by state)
        - conclusion: str
    """
    # State distribution
    state_counts = events_with_state['Cromwell_Tap_State'].value_counts()
    state_distribution = Counter(state_counts.to_dict())

    total_events = len(events_with_state)
    mentions = events_with_state['Cromwell_Mentioned'].sum()

    events_when_open = state_counts.get('open', 0)
    events_when_closed = state_counts.get('closed', 0)
    events_when_unclear = state_counts.get('mentioned_unclear', 0)
    events_not_mentioned = state_counts.get('not_mentioned', 0)

    # Calculate time span for rate analysis
    time_span_days = (events_with_state[datetime_col].max() -
                      events_with_state[datetime_col].min()).days

    if time_span_days == 0:
        time_span_days = 1

    # Event rates (events per day) - rough approximation
    # Note: This assumes states are distributed over time, which may not be accurate
    rate_when_open = events_when_open / (time_span_days * (events_when_open / total_events)) if events_when_open > 0 else 0
    rate_when_closed = events_when_closed / (time_span_days * (events_when_closed / total_events)) if events_when_closed > 0 else 0
    rate_not_mentioned = events_not_mentioned / (time_span_days * (events_not_mentioned / total_events)) if events_not_mentioned > 0 else 0

    # Simplified conclusion based on state distribution
    if mentions > 0:
        if events_when_open > events_when_closed:
            conclusion = f"Cromwell Tap 31 more frequently mentioned as 'open' ({events_when_open} events) than 'closed' ({events_when_closed} events)"
        elif events_when_closed > events_when_open:
            conclusion = f"Cromwell Tap 31 more frequently mentioned as 'closed' ({events_when_closed} events) than 'open' ({events_when_open} events)"
        else:
            conclusion = f"Cromwell Tap 31 mentioned equally in 'open' and 'closed' states"
    else:
        conclusion = "Cromwell Tap 31 not mentioned in any event descriptions"

    return {
        'state_distribution': state_distribution,
        'mentions': int(mentions),
        'events_when_open': int(events_when_open),
        'events_when_closed': int(events_when_closed),
        'events_when_unclear': int(events_when_unclear),
        'events_not_mentioned': int(events_not_mentioned),
        'pct_when_open': (events_when_open / total_events * 100) if total_events > 0 else 0,
        'pct_when_closed': (events_when_closed / total_events * 100) if total_events > 0 else 0,
        'rate_analysis': {
            'note': 'Rate estimates are approximate based on state distribution',
            'events_per_day_when_open': rate_when_open,
            'events_per_day_when_closed': rate_when_closed,
            'events_per_day_not_mentioned': rate_not_mentioned
        },
        'conclusion': conclusion
    }


def extract_network_interconnection_context(section150_events: pd.DataFrame,
                                             cause_col: str) -> Dict:
    """
    Text-mine for network interconnection keywords to infer criticality.

    Keywords searched:
    - "PSO" (Public Service Oklahoma) - indicates cross-utility coordination
    - "OMPA" (Oklahoma Municipal Power Authority) - municipal coordination
    - "DCC notified" - dispatch center coordination
    - "Weleetka" / "Wetumka" - specific interconnection points

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events
    cause_col : str
        Name of cause column

    Returns:
    --------
    Dict with:
        - pso_mentions: int
        - ompa_mentions: int
        - dcc_mentions: int
        - weleetka_mentions: int
        - wetumka_mentions: int
        - interconnection_score: float (weighted sum)
        - high_coordination_events: int (events requiring multi-party coordination)
        - coordination_keywords: Counter (all keywords found)
    """
    # Define keywords and their weights
    keywords = {
        'PSO': 1.0,
        'OMPA': 0.8,
        'DCC': 0.6,
        'Weleetka': 0.5,
        'Weleekta': 0.5,  # Alternative spelling
        'Wetumka': 0.5
    }

    # Count mentions
    keyword_counts = {}
    for keyword in keywords.keys():
        count = section150_events[cause_col].str.contains(
            keyword, case=False, na=False, regex=False
        ).sum()
        keyword_counts[keyword] = int(count)

    # Combine alternative spellings
    weleetka_total = keyword_counts.get('Weleetka', 0) + keyword_counts.get('Weleekta', 0)

    # Calculate interconnection score (weighted sum normalized by total events)
    total_events = len(section150_events)
    interconnection_score = 0.0
    for keyword, weight in keywords.items():
        interconnection_score += (keyword_counts.get(keyword, 0) / total_events) * weight

    # High coordination events (mention 2+ entities)
    def count_entities(text):
        if pd.isna(text):
            return 0
        text_lower = str(text).lower()
        count = 0
        for keyword in keywords.keys():
            if keyword.lower() in text_lower:
                count += 1
        return count

    entities_per_event = section150_events[cause_col].apply(count_entities)
    high_coordination_events = (entities_per_event >= 2).sum()

    return {
        'pso_mentions': keyword_counts.get('PSO', 0),
        'ompa_mentions': keyword_counts.get('OMPA', 0),
        'dcc_mentions': keyword_counts.get('DCC', 0),
        'weleetka_mentions': weleetka_total,
        'wetumka_mentions': keyword_counts.get('Wetumka', 0),
        'interconnection_score': interconnection_score,
        'high_coordination_events': int(high_coordination_events),
        'coordination_keywords': Counter(keyword_counts),
        'interpretation': f"{keyword_counts.get('PSO', 0)} PSO mentions, {keyword_counts.get('OMPA', 0)} OMPA mentions suggest high inter-utility coordination"
    }


def compare_equipment_age(pmu_df: pd.DataFrame,
                          disturbance_df: pd.DataFrame) -> Dict:
    """
    Compare Section 150 equipment age to similar sections.

    Uses InService dates from PMU data to:
    - Calculate Age_Years for Section 150 and all sections
    - Find similar sections (same voltage & PMU type)
    - Statistical comparison (t-test)
    - Correlation: age vs event frequency

    Parameters:
    -----------
    pmu_df : pd.DataFrame
        PMU metadata
    disturbance_df : pd.DataFrame
        All disturbance events

    Returns:
    --------
    Dict with:
        - section150_age_years: float
        - section150_inservice_date: Timestamp
        - similar_sections_mean_age: float
        - similar_sections_std_age: float
        - similar_sections_count: int
        - age_percentile: float (Section 150 rank among similar)
        - age_correlation_with_events: float (Pearson r)
        - t_statistic: float
        - p_value: float
        - conclusion: str
    """
    # Get Section 150 details
    sec150_pmu = pmu_df[pmu_df['SectionID'] == cfg.TARGET_SECTION_ID]

    if sec150_pmu.empty or 'InService' not in pmu_df.columns:
        return {
            'error': 'No PMU data or InService date not available',
            'conclusion': 'Cannot perform age analysis - missing data'
        }

    # Calculate ages for all sections
    pmu_with_age = pmu_df.copy()
    pmu_with_age['Age_Years'] = (pd.Timestamp.now() - pmu_with_age['InService']).dt.days / 365.25

    # Get Section 150 age
    sec150_age = pmu_with_age[pmu_with_age['SectionID'] == cfg.TARGET_SECTION_ID]['Age_Years'].values[0]
    sec150_inservice = sec150_pmu['InService'].values[0]

    # Identify voltage and type columns
    voltage_col = None
    for col in pmu_df.columns:
        if 'volt' in col.lower() or 'kv' in col.lower():
            voltage_col = col
            break

    type_col = None
    for col in pmu_df.columns:
        if 'type' in col.lower() and 'pmu' in col.lower():
            type_col = col
            break

    # Find similar sections
    if voltage_col and type_col:
        sec150_voltage = sec150_pmu[voltage_col].values[0]
        sec150_type = sec150_pmu[type_col].values[0]

        similar_mask = (
            (pmu_with_age[voltage_col] == sec150_voltage) &
            (pmu_with_age[type_col] == sec150_type) &
            (pmu_with_age['SectionID'] != cfg.TARGET_SECTION_ID)
        )
        similar_sections = pmu_with_age[similar_mask]
    else:
        similar_sections = pmu_with_age[pmu_with_age['SectionID'] != cfg.TARGET_SECTION_ID]

    similar_ages = similar_sections['Age_Years'].dropna()

    # Statistics
    similar_mean_age = similar_ages.mean()
    similar_std_age = similar_ages.std()
    similar_count = len(similar_ages)

    # Age percentile among similar sections
    age_percentile = (similar_ages < sec150_age).sum() / len(similar_ages) * 100 if len(similar_ages) > 0 else 50

    # Correlation with event count
    event_counts = disturbance_df.groupby('SectionID').size()
    age_event_df = pmu_with_age[['SectionID', 'Age_Years']].copy()
    age_event_df['Event_Count'] = age_event_df['SectionID'].map(event_counts).fillna(0)
    age_event_df = age_event_df.dropna()

    if len(age_event_df) > 2:
        age_corr, age_corr_p = stats.pearsonr(age_event_df['Age_Years'], age_event_df['Event_Count'])
    else:
        age_corr, age_corr_p = None, None

    # T-test: Is Section 150 age significantly different from similar sections?
    if len(similar_ages) > 1:
        t_stat, p_value = stats.ttest_1samp(similar_ages, sec150_age)
    else:
        t_stat, p_value = None, None

    # Conclusion
    if age_percentile > 75:
        age_desc = f"older than {age_percentile:.0f}% of similar sections"
    elif age_percentile < 25:
        age_desc = f"newer than {100-age_percentile:.0f}% of similar sections"
    else:
        age_desc = "average age compared to similar sections"

    if p_value and p_value < 0.05:
        significance = "significantly different from"
    else:
        significance = "not significantly different from"

    conclusion = f"Section 150 is {sec150_age:.1f} years old ({age_desc}), {significance} the mean age of similar sections ({similar_mean_age:.1f} years)"

    return {
        'section150_age_years': float(sec150_age),
        'section150_inservice_date': pd.Timestamp(sec150_inservice),
        'similar_sections_mean_age': float(similar_mean_age),
        'similar_sections_std_age': float(similar_std_age),
        'similar_sections_count': int(similar_count),
        'age_percentile': float(age_percentile),
        'age_correlation_with_events': float(age_corr) if age_corr else None,
        'age_correlation_p_value': float(age_corr_p) if age_corr_p else None,
        't_statistic': float(t_stat) if t_stat else None,
        'p_value': float(p_value) if p_value else None,
        'conclusion': conclusion
    }


def get_operational_context_summary(section150_events: pd.DataFrame,
                                     pmu_df: pd.DataFrame,
                                     disturbance_df: pd.DataFrame,
                                     network_baselines: Dict) -> Dict:
    """
    Complete operational context analysis for Section 150.

    Combines all operational context analyses:
    - Cause field parsing
    - Cromwell Tap 31 state extraction and correlation
    - Network interconnection context
    - Equipment age comparison

    Parameters:
    -----------
    section150_events : pd.DataFrame
        Section 150 events
    pmu_df : pd.DataFrame
        PMU metadata
    disturbance_df : pd.DataFrame
        All disturbance events
    network_baselines : Dict
        Network baseline statistics

    Returns:
    --------
    Dict with keys:
        - events_with_context: DataFrame (events with new columns)
        - cromwell: Dict (Cromwell Tap analysis)
        - interconnection: Dict (network interconnection analysis)
        - equipment_age: Dict (age comparison)
    """
    cause_col = network_baselines['cause_col']
    datetime_col = network_baselines['datetime_col']

    # Extract Cromwell Tap state
    events_with_context = extract_cromwell_tap_state(section150_events, cause_col)

    # Cromwell Tap correlation analysis
    cromwell_analysis = analyze_cromwell_tap_correlation(events_with_context, datetime_col)

    # Network interconnection context
    interconnection_analysis = extract_network_interconnection_context(section150_events, cause_col)

    # Equipment age comparison
    equipment_age_analysis = compare_equipment_age(pmu_df, disturbance_df)

    return {
        'events_with_context': events_with_context,
        'cromwell': cromwell_analysis,
        'interconnection': interconnection_analysis,
        'equipment_age': equipment_age_analysis
    }
