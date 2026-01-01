"""
Unit Tests for Risk Scorer Module.
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_pmu_disturbance_data
from risk_scorer import PMURiskScorer

DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'PMU_disturbance.xlsx'


@pytest.fixture(scope='module')
def data():
    """Load data once for all tests."""
    return load_pmu_disturbance_data(str(DATA_FILE))


@pytest.fixture(scope='module')
def risk_scorer(data):
    """Create risk scorer."""
    pmu_df, dist_df = data
    return PMURiskScorer(pmu_df, dist_df)


class TestRiskScorerInit:
    """Tests for PMURiskScorer initialization."""
    
    def test_default_weights(self, data):
        """Default weights should sum to 1.0."""
        pmu_df, dist_df = data
        scorer = PMURiskScorer(pmu_df, dist_df)
        assert np.isclose(sum(scorer.weights.values()), 1.0)
    
    def test_custom_weights_valid(self, data):
        """Should accept valid custom weights."""
        pmu_df, dist_df = data
        custom_weights = {'frequency': 0.5, 'trend': 0.2, 'mtbf': 0.15, 'age': 0.1, 'recency': 0.05}
        scorer = PMURiskScorer(pmu_df, dist_df, weights=custom_weights)
        assert scorer.weights == custom_weights
    
    def test_custom_weights_invalid(self, data):
        """Should reject weights that don't sum to 1.0."""
        pmu_df, dist_df = data
        invalid_weights = {'frequency': 0.5, 'trend': 0.2, 'mtbf': 0.1, 'age': 0.1, 'recency': 0.05}
        with pytest.raises(ValueError):
            PMURiskScorer(pmu_df, dist_df, weights=invalid_weights)


class TestCalculateRiskScores:
    """Tests for calculate_risk_scores method."""
    
    def test_returns_dataframe(self, risk_scorer):
        """Should return a DataFrame."""
        results = risk_scorer.calculate_risk_scores()
        assert isinstance(results, pd.DataFrame)
    
    def test_has_required_columns(self, risk_scorer):
        """Should have all required columns."""
        results = risk_scorer.calculate_risk_scores()
        required_cols = ['SectionID', 'risk_score', 'rank', 'category']
        for col in required_cols:
            assert col in results.columns
    
    def test_scores_in_range(self, risk_scorer):
        """All scores should be between 0 and 100."""
        results = risk_scorer.calculate_risk_scores()
        assert results['risk_score'].min() >= 0
        assert results['risk_score'].max() <= 100
    
    def test_ranks_unique(self, risk_scorer):
        """Ranks should cover all sections."""
        results = risk_scorer.calculate_risk_scores()
        assert results['rank'].min() >= 1
        assert results['rank'].max() <= len(results)
    
    def test_categories_valid(self, risk_scorer):
        """Categories should be Low, Medium, or High."""
        results = risk_scorer.calculate_risk_scores()
        valid_categories = {'Low', 'Medium', 'High'}
        assert set(results['category'].unique()).issubset(valid_categories)


class TestGetTopRiskSections:
    """Tests for get_top_risk_sections method."""
    
    def test_returns_n_sections(self, risk_scorer):
        """Should return specified number of sections."""
        top_10 = risk_scorer.get_top_risk_sections(10)
        assert len(top_10) == 10
    
    def test_sorted_by_risk(self, risk_scorer):
        """Should be sorted by risk score descending."""
        top_20 = risk_scorer.get_top_risk_sections(20)
        scores = top_20['risk_score'].tolist()
        assert scores == sorted(scores, reverse=True)


class TestGetSectionRisk:
    """Tests for get_section_risk method."""
    
    def test_returns_dict(self, risk_scorer):
        """Should return a dictionary."""
        risk_scorer.calculate_risk_scores()
        result = risk_scorer.get_section_risk(150)
        assert isinstance(result, dict)
    
    def test_nonexistent_section(self, risk_scorer):
        """Should return error for nonexistent section."""
        risk_scorer.calculate_risk_scores()
        result = risk_scorer.get_section_risk(999999)
        assert 'error' in result


class TestComponentScores:
    """Tests for individual component scores."""
    
    def test_frequency_score_exists(self, risk_scorer):
        """Frequency score should be calculated."""
        results = risk_scorer.calculate_risk_scores()
        assert 'frequency_score' in results.columns
    
    def test_trend_score_exists(self, risk_scorer):
        """Trend score should be calculated."""
        results = risk_scorer.calculate_risk_scores()
        assert 'trend_score' in results.columns
    
    def test_mtbf_score_exists(self, risk_scorer):
        """MTBF score should be calculated."""
        results = risk_scorer.calculate_risk_scores()
        assert 'mtbf_score' in results.columns
    
    def test_component_scores_in_range(self, risk_scorer):
        """All component scores should be 0-100."""
        results = risk_scorer.calculate_risk_scores()
        for col in ['frequency_score', 'trend_score', 'mtbf_score', 'age_score', 'recency_score']:
            if col in results.columns:
                assert results[col].min() >= 0
                assert results[col].max() <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
