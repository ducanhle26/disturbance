"""
Risk Scoring Module for PMU Reliability Framework.

Implements multi-dimensional risk scoring combining frequency, trend, MTBF, age, and recency.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats

try:
    from .data_loader import get_section_events, calculate_event_statistics, _find_datetime_column
except ImportError:
    from data_loader import get_section_events, calculate_event_statistics, _find_datetime_column


class PMURiskScorer:
    """
    Multi-dimensional risk scorer for PMU sections.
    
    Calculates composite risk scores based on:
    - Frequency: Event count (35% default weight)
    - Trend: Are events increasing over time? (25% default weight)
    - MTBF: Mean time between failures - inverted (20% default weight)
    - Age: Equipment age (10% default weight)
    - Recency: Time since last event - inverted (10% default weight)
    """
    
    DEFAULT_WEIGHTS = {
        'frequency': 0.35,
        'trend': 0.25,
        'mtbf': 0.20,
        'age': 0.10,
        'recency': 0.10
    }
    
    def __init__(self, pmu_df: pd.DataFrame, dist_df: pd.DataFrame, 
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize risk scorer.
        
        Parameters
        ----------
        pmu_df : pd.DataFrame
            PMU installations dataframe
        dist_df : pd.DataFrame
            Disturbance events dataframe
        weights : Dict[str, float], optional
            Custom weights for risk components (must sum to 1.0)
        """
        self.pmu_df = pmu_df.copy()
        self.dist_df = dist_df.copy()
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        if not np.isclose(sum(self.weights.values()), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights.values())}")
        
        self.datetime_col = _find_datetime_column(dist_df)
        self.results_df = None
        
    def calculate_risk_scores(self) -> pd.DataFrame:
        """
        Calculate risk scores for all sections.
        
        Returns
        -------
        pd.DataFrame
            Results with columns: SectionID, risk_score, rank, category, 
            and individual component scores
        """
        sections = self.pmu_df['SectionID'].unique()
        
        raw_scores = []
        for section_id in sections:
            scores = self._calculate_section_scores(section_id)
            raw_scores.append(scores)
        
        results = pd.DataFrame(raw_scores)
        
        for component in ['frequency', 'trend', 'mtbf', 'age', 'recency']:
            col = f'{component}_raw'
            if col in results.columns:
                results[f'{component}_score'] = self._normalize_score(
                    results[col], 
                    invert=(component in ['mtbf', 'recency'])
                )
        
        results['risk_score'] = (
            results['frequency_score'] * self.weights['frequency'] +
            results['trend_score'] * self.weights['trend'] +
            results['mtbf_score'] * self.weights['mtbf'] +
            results['age_score'] * self.weights['age'] +
            results['recency_score'] * self.weights['recency']
        )
        
        results['rank'] = results['risk_score'].rank(ascending=False, method='min').astype(int)
        results['category'] = results['risk_score'].apply(self._categorize_risk)
        results = results.sort_values('rank')
        
        self.results_df = results
        return results
    
    def _calculate_section_scores(self, section_id: int) -> Dict:
        """Calculate raw scores for a single section."""
        events = get_section_events(self.dist_df, section_id)
        stats = calculate_event_statistics(events)
        pmu_info = self.pmu_df[self.pmu_df['SectionID'] == section_id]
        
        frequency = stats['count']
        
        trend = 0.0
        if len(events) >= 3 and self.datetime_col:
            trend = self._calculate_trend(events)
        
        mtbf = stats['mtbf_days'] if stats['mtbf_days'] else np.inf
        
        age = 0.0
        if 'Age_Years' in pmu_info.columns and len(pmu_info) > 0:
            age = pmu_info['Age_Years'].iloc[0]
            if pd.isna(age):
                age = 0.0
        
        recency = np.inf
        if self.datetime_col and len(events) > 0:
            last_event = events[self.datetime_col].max()
            recency = (pd.Timestamp.now() - last_event).days
        
        return {
            'SectionID': section_id,
            'event_count': frequency,
            'frequency_raw': frequency,
            'trend_raw': trend,
            'mtbf_raw': mtbf,
            'age_raw': age,
            'recency_raw': recency,
            'mtbf_days': stats['mtbf_days'],
            'first_event': stats['first_event'],
            'last_event': stats['last_event']
        }
    
    def _calculate_trend(self, events: pd.DataFrame) -> float:
        """Calculate trend score using linear regression on event frequency."""
        if self.datetime_col is None:
            return 0.0
        
        events = events.copy()
        events['date'] = events[self.datetime_col].dt.date
        daily_counts = events.groupby('date').size().reset_index(name='count')
        
        if len(daily_counts) < 2:
            return 0.0
        
        daily_counts['day_num'] = (pd.to_datetime(daily_counts['date']) - 
                                    pd.to_datetime(daily_counts['date'].min())).dt.days
        
        slope, _, r_value, _, _ = stats.linregress(daily_counts['day_num'], daily_counts['count'])
        
        return slope * 1000  # Scale for visibility
    
    def _normalize_score(self, series: pd.Series, invert: bool = False) -> pd.Series:
        """Normalize scores to 0-100 scale using min-max normalization."""
        finite_values = series.replace([np.inf, -np.inf], np.nan)
        
        min_val = finite_values.min()
        max_val = finite_values.max()
        
        if max_val == min_val:
            normalized = pd.Series([50.0] * len(series), index=series.index)
        else:
            normalized = (finite_values - min_val) / (max_val - min_val) * 100
        
        normalized = normalized.fillna(0 if invert else 100)
        
        if invert:
            normalized = 100 - normalized
        
        return normalized.clip(0, 100)
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk score into Low/Medium/High."""
        if score >= 66.67:
            return 'High'
        elif score >= 33.33:
            return 'Medium'
        else:
            return 'Low'
    
    def get_top_risk_sections(self, n: int = 20) -> pd.DataFrame:
        """Get the top N highest-risk sections."""
        if self.results_df is None:
            self.calculate_risk_scores()
        return self.results_df.head(n)
    
    def get_section_risk(self, section_id: int) -> Dict:
        """Get detailed risk information for a specific section."""
        if self.results_df is None:
            self.calculate_risk_scores()
        
        section_data = self.results_df[self.results_df['SectionID'] == section_id]
        if len(section_data) == 0:
            return {'error': f'Section {section_id} not found'}
        
        return section_data.iloc[0].to_dict()
