"""
Causality and pattern mining utilities for PMU disturbance analysis.
Includes cause analysis, association rules, sequential patterns, and reliability metrics.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


def analyze_cause_distribution(df: pd.DataFrame,
                               cause_col: str = 'Cause') -> pd.DataFrame:
    """
    Analyze frequency distribution of disturbance causes.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    cause_col : str
        Column name containing causes

    Returns:
    --------
    pd.DataFrame
        Cause frequency with counts and percentages
    """
    cause_counts = df[cause_col].value_counts()
    cause_pct = (cause_counts / len(df) * 100).round(2)

    result = pd.DataFrame({
        'Count': cause_counts,
        'Percentage': cause_pct,
        'Cumulative_Percentage': cause_pct.cumsum()
    })

    return result


def calculate_pareto_80_20(cause_dist: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Identify causes responsible for 80% of disturbances (Pareto principle).

    Parameters:
    -----------
    cause_dist : pd.DataFrame
        Output from analyze_cause_distribution

    Returns:
    --------
    Tuple[pd.DataFrame, int]
        (Top causes contributing to 80%, number of causes)
    """
    pareto_causes = cause_dist[cause_dist['Cumulative_Percentage'] <= 80]

    return pareto_causes, len(pareto_causes)


def mine_association_rules(df: pd.DataFrame,
                           cause_col: str,
                           section_col: str = 'SectionID',
                           min_support: float = 0.01,
                           min_confidence: float = 0.5) -> pd.DataFrame:
    """
    Mine association rules between disturbance causes using Apriori algorithm.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    cause_col : str
        Column containing causes
    section_col : str
        Column to group transactions by
    min_support : float
        Minimum support threshold
    min_confidence : float
        Minimum confidence threshold

    Returns:
    --------
    pd.DataFrame
        Association rules
    """
    # Create transactions (causes per section)
    transactions = df.groupby(section_col)[cause_col].apply(list).values

    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        return pd.DataFrame()

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)

    # Sort by confidence and lift
    rules = rules.sort_values(['confidence', 'lift'], ascending=False)

    return rules


def create_cooccurrence_matrix(df: pd.DataFrame,
                               cause_col: str,
                               section_col: str = 'SectionID') -> pd.DataFrame:
    """
    Create co-occurrence matrix showing which causes appear together.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    cause_col : str
        Column containing causes
    section_col : str
        Column to group by

    Returns:
    --------
    pd.DataFrame
        Co-occurrence matrix
    """
    # Get unique causes
    causes = df[cause_col].unique()

    # Initialize matrix
    matrix = pd.DataFrame(0, index=causes, columns=causes)

    # Count co-occurrences by section
    for section_id, group in df.groupby(section_col):
        section_causes = group[cause_col].unique()
        for c1 in section_causes:
            for c2 in section_causes:
                matrix.loc[c1, c2] += 1

    return matrix


def detect_sequential_patterns(df: pd.DataFrame,
                               datetime_col: str,
                               cause_col: str,
                               section_col: str = 'SectionID',
                               window_days: int = 7) -> pd.DataFrame:
    """
    Detect sequential patterns: Cause A followed by Cause B within time window.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    datetime_col : str
        Datetime column
    cause_col : str
        Cause column
    section_col : str
        Section identifier
    window_days : int
        Time window in days

    Returns:
    --------
    pd.DataFrame
        Sequential patterns with counts
    """
    df = df.sort_values([section_col, datetime_col])

    sequences = []

    for section_id, group in df.groupby(section_col):
        group = group.sort_values(datetime_col).reset_index(drop=True)

        for i in range(len(group) - 1):
            current_cause = group.loc[i, cause_col]
            current_time = group.loc[i, datetime_col]

            # Look ahead within window
            for j in range(i + 1, len(group)):
                next_cause = group.loc[j, cause_col]
                next_time = group.loc[j, datetime_col]

                time_diff = (next_time - current_time).total_seconds() / (24 * 3600)

                if time_diff <= window_days:
                    sequences.append({
                        'From': current_cause,
                        'To': next_cause,
                        'Days_Apart': time_diff
                    })
                else:
                    break

    if not sequences:
        return pd.DataFrame()

    seq_df = pd.DataFrame(sequences)

    # Aggregate patterns
    pattern_counts = seq_df.groupby(['From', 'To']).agg({
        'Days_Apart': ['count', 'mean', 'std']
    }).reset_index()

    pattern_counts.columns = ['From', 'To', 'Count', 'Avg_Days', 'Std_Days']
    pattern_counts = pattern_counts.sort_values('Count', ascending=False)

    return pattern_counts


def calculate_mtbf_mttr(df: pd.DataFrame,
                        datetime_col: str,
                        section_col: str = 'SectionID',
                        operations_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate Mean Time Between Failures (MTBF) and Mean Time To Repair (MTTR) per section.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    datetime_col : str
        Datetime column
    section_col : str
        Section identifier
    operations_col : str, optional
        Column containing operational duration

    Returns:
    --------
    pd.DataFrame
        MTBF and MTTR statistics per section
    """
    results = []

    for section_id, group in df.groupby(section_col):
        group = group.sort_values(datetime_col)

        # Calculate MTBF (time between failures)
        if len(group) > 1:
            time_diffs = group[datetime_col].diff().dt.total_seconds() / 3600  # hours
            mtbf = time_diffs.mean()
            mtbf_std = time_diffs.std()
        else:
            mtbf = np.nan
            mtbf_std = np.nan

        # Calculate MTTR if operations data available
        if operations_col and operations_col in df.columns:
            mttr = group[operations_col].mean()
            mttr_std = group[operations_col].std()
        else:
            mttr = np.nan
            mttr_std = np.nan

        results.append({
            'SectionID': section_id,
            'Failure_Count': len(group),
            'MTBF_hours': mtbf,
            'MTBF_std': mtbf_std,
            'MTTR_hours': mttr,
            'MTTR_std': mttr_std,
            'Failure_Rate': 1 / mtbf if not np.isnan(mtbf) and mtbf > 0 else np.nan
        })

    return pd.DataFrame(results).sort_values('Failure_Count', ascending=False)


def calculate_cause_severity(df: pd.DataFrame,
                             cause_col: str,
                             operations_col: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate average severity/impact by cause type.

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    cause_col : str
        Cause column
    operations_col : str, optional
        Column indicating impact/duration

    Returns:
    --------
    pd.DataFrame
        Severity metrics by cause
    """
    severity_metrics = df.groupby(cause_col).agg({
        cause_col: 'count'
    }).rename(columns={cause_col: 'Frequency'})

    if operations_col and operations_col in df.columns:
        ops_stats = df.groupby(cause_col)[operations_col].agg(['mean', 'std', 'max'])
        severity_metrics = severity_metrics.join(ops_stats)

    severity_metrics = severity_metrics.sort_values('Frequency', ascending=False)

    return severity_metrics


def create_transition_matrix(df: pd.DataFrame,
                             datetime_col: str,
                             cause_col: str,
                             section_col: str = 'SectionID') -> pd.DataFrame:
    """
    Create transition probability matrix: P(Cause B | Cause A).

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    datetime_col : str
        Datetime column
    cause_col : str
        Cause column
    section_col : str
        Section identifier

    Returns:
    --------
    pd.DataFrame
        Transition probability matrix
    """
    transitions = defaultdict(Counter)

    for section_id, group in df.groupby(section_col):
        group = group.sort_values(datetime_col)
        causes = group[cause_col].tolist()

        for i in range(len(causes) - 1):
            current = causes[i]
            next_cause = causes[i + 1]
            transitions[current][next_cause] += 1

    # Get all unique causes
    all_causes = sorted(df[cause_col].unique())

    # Create matrix
    matrix = pd.DataFrame(0.0, index=all_causes, columns=all_causes)

    for current, next_dict in transitions.items():
        total = sum(next_dict.values())
        for next_cause, count in next_dict.items():
            matrix.loc[current, next_cause] = count / total

    return matrix
