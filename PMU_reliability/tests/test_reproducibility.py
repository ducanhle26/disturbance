"""
Reproducibility Tests for PMU Reliability Framework.

Validates that Section 150 analysis matches original findings.
"""

import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_pmu_disturbance_data, get_section_events, calculate_event_statistics
from risk_scorer import PMURiskScorer
from temporal_analysis import TemporalAnalyzer

DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'PMU_disturbance.xlsx'
SECTION_150_ID = 150


@pytest.fixture(scope='module')
def data():
    """Load data once for all tests."""
    pmu_df, dist_df = load_pmu_disturbance_data(str(DATA_FILE))
    return pmu_df, dist_df


@pytest.fixture(scope='module')
def section_150_events(data):
    """Get Section 150 events."""
    _, dist_df = data
    return get_section_events(dist_df, SECTION_150_ID)


@pytest.fixture(scope='module')
def risk_scorer(data):
    """Create risk scorer."""
    pmu_df, dist_df = data
    scorer = PMURiskScorer(pmu_df, dist_df)
    scorer.calculate_risk_scores()
    return scorer


class TestSection150EventCount:
    """Test Section 150 event count matches expected 301."""
    
    def test_section_150_event_count(self, section_150_events):
        """Section 150 should have exactly 301 events."""
        assert len(section_150_events) == 301, \
            f"Expected 301 events, got {len(section_150_events)}"


class TestSection150MTBF:
    """Test Section 150 MTBF matches expected ~16.4 days."""
    
    def test_section_150_mtbf(self, section_150_events):
        """Section 150 MTBF should be approximately 16.4 days."""
        stats = calculate_event_statistics(section_150_events)
        mtbf = stats['mtbf_days']
        
        assert mtbf is not None, "MTBF should not be None"
        assert 15.5 <= mtbf <= 17.5, \
            f"Expected MTBF ~16.4 days, got {mtbf:.2f}"


class TestSection150NetworkRatio:
    """Test Section 150 network ratio matches expected ~25x."""
    
    def test_section_150_network_ratio(self, data, section_150_events):
        """Section 150 should have ~25x the network average events."""
        pmu_df, dist_df = data
        
        section_150_count = len(section_150_events)
        total_sections = dist_df['SectionID'].nunique()
        network_avg = len(dist_df) / total_sections
        
        ratio = section_150_count / network_avg
        
        assert 20 <= ratio <= 30, \
            f"Expected ratio ~25x, got {ratio:.1f}x"


class TestSection150RiskRank:
    """Test Section 150 ranks #1 in risk scoring."""
    
    def test_section_150_risk_rank(self, risk_scorer):
        """Section 150 should rank #1 in risk scoring."""
        section_risk = risk_scorer.get_section_risk(SECTION_150_ID)
        
        assert 'error' not in section_risk, f"Section 150 not found: {section_risk}"
        assert section_risk['rank'] == 1, \
            f"Expected rank 1, got {section_risk['rank']}"


class TestSection150TopCause:
    """Test Section 150 top cause is Unknown."""
    
    def test_section_150_top_cause(self, section_150_events):
        """Top cause for Section 150 should be Unknown with ~51 events."""
        cause_col = None
        for col in section_150_events.columns:
            if 'cause' in col.lower():
                cause_col = col
                break
        
        if cause_col is None:
            pytest.skip("No cause column found")
        
        cause_counts = section_150_events[cause_col].value_counts()
        top_cause = cause_counts.index[0]
        top_count = cause_counts.iloc[0]
        
        assert 'unknown' in top_cause.lower(), \
            f"Expected top cause 'Unknown', got '{top_cause}'"
        assert 45 <= top_count <= 60, \
            f"Expected ~51 events, got {top_count}"


class TestSection150PeakHour:
    """Test Section 150 peak hour is 19:00."""
    
    def test_section_150_peak_hour(self, section_150_events):
        """Section 150 peak hour should be 19:00 (7 PM)."""
        analyzer = TemporalAnalyzer(section_150_events)
        peaks = analyzer.calculate_peak_periods()
        
        assert peaks['peak_hour'] == 19, \
            f"Expected peak hour 19, got {peaks['peak_hour']}"


class TestSection150RiskScore:
    """Test Section 150 risk score is in expected range."""
    
    def test_section_150_risk_score(self, risk_scorer):
        """Section 150 risk score should be in 60-70 range."""
        section_risk = risk_scorer.get_section_risk(SECTION_150_ID)
        score = section_risk['risk_score']
        
        assert 55 <= score <= 75, \
            f"Expected risk score 60-70, got {score:.1f}"


class TestDataIntegrity:
    """Test data loading integrity."""
    
    def test_total_pmu_count(self, data):
        """Should load 533 PMUs."""
        pmu_df, _ = data
        assert len(pmu_df) == 533, f"Expected 533 PMUs, got {len(pmu_df)}"
    
    def test_total_event_count(self, data):
        """Should load 9369 disturbance events."""
        _, dist_df = data
        assert len(dist_df) == 9369, f"Expected 9369 events, got {len(dist_df)}"
    
    def test_section_id_link(self, data):
        """SectionID should exist in both dataframes."""
        pmu_df, dist_df = data
        assert 'SectionID' in pmu_df.columns
        assert 'SectionID' in dist_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
