"""
Unit Tests for Temporal Analysis Module.
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_pmu_disturbance_data, get_section_events
from temporal_analysis import TemporalAnalyzer

DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'PMU_disturbance.xlsx'


@pytest.fixture(scope='module')
def data():
    """Load data once for all tests."""
    return load_pmu_disturbance_data(str(DATA_FILE))


@pytest.fixture(scope='module')
def section_150_events(data):
    """Get Section 150 events."""
    _, dist_df = data
    return get_section_events(dist_df, 150)


@pytest.fixture(scope='module')
def analyzer(section_150_events):
    """Create temporal analyzer."""
    return TemporalAnalyzer(section_150_events)


class TestTemporalAnalyzerInit:
    """Tests for TemporalAnalyzer initialization."""
    
    def test_creates_analyzer(self, section_150_events):
        """Should create analyzer from events."""
        analyzer = TemporalAnalyzer(section_150_events)
        assert analyzer is not None
    
    def test_detects_datetime_column(self, analyzer):
        """Should detect datetime column."""
        assert analyzer.datetime_col is not None


class TestDetectAnomalies:
    """Tests for detect_anomalies method."""
    
    def test_iqr_method(self, analyzer):
        """IQR method should return DataFrame."""
        result = analyzer.detect_anomalies(method='iqr')
        assert isinstance(result, pd.DataFrame)
    
    def test_zscore_method(self, analyzer):
        """Z-score method should return DataFrame."""
        result = analyzer.detect_anomalies(method='zscore')
        assert isinstance(result, pd.DataFrame)
    
    def test_invalid_method(self, analyzer):
        """Invalid method should raise error."""
        with pytest.raises(ValueError):
            analyzer.detect_anomalies(method='invalid')


class TestHourlyPattern:
    """Tests for calculate_hourly_pattern method."""
    
    def test_returns_series(self, analyzer):
        """Should return a Series."""
        result = analyzer.calculate_hourly_pattern()
        assert isinstance(result, pd.Series)
    
    def test_has_24_hours(self, analyzer):
        """Should have 24 hours (0-23)."""
        result = analyzer.calculate_hourly_pattern()
        assert len(result) == 24
        assert list(result.index) == list(range(24))
    
    def test_non_negative_counts(self, analyzer):
        """All counts should be non-negative."""
        result = analyzer.calculate_hourly_pattern()
        assert all(result >= 0)


class TestDailyPattern:
    """Tests for calculate_daily_pattern method."""
    
    def test_returns_series(self, analyzer):
        """Should return a Series."""
        result = analyzer.calculate_daily_pattern()
        assert isinstance(result, pd.Series)
    
    def test_has_7_days(self, analyzer):
        """Should have 7 days."""
        result = analyzer.calculate_daily_pattern()
        assert len(result) == 7


class TestMonthlyPattern:
    """Tests for calculate_monthly_pattern method."""
    
    def test_returns_series(self, analyzer):
        """Should return a Series."""
        result = analyzer.calculate_monthly_pattern()
        assert isinstance(result, pd.Series)
    
    def test_has_12_months(self, analyzer):
        """Should have 12 months."""
        result = analyzer.calculate_monthly_pattern()
        assert len(result) == 12


class TestPeakPeriods:
    """Tests for calculate_peak_periods method."""
    
    def test_returns_dict(self, analyzer):
        """Should return a dictionary."""
        result = analyzer.calculate_peak_periods()
        assert isinstance(result, dict)
    
    def test_has_required_keys(self, analyzer):
        """Should have all required keys."""
        result = analyzer.calculate_peak_periods()
        required_keys = ['peak_hour', 'peak_day', 'peak_month']
        for key in required_keys:
            assert key in result
    
    def test_peak_hour_valid(self, analyzer):
        """Peak hour should be 0-23."""
        result = analyzer.calculate_peak_periods()
        assert 0 <= result['peak_hour'] <= 23


class TestClustering:
    """Tests for test_clustering method."""
    
    def test_returns_dict(self, analyzer):
        """Should return a dictionary."""
        result = analyzer.test_clustering()
        assert isinstance(result, dict)
    
    def test_has_dispersion_index(self, analyzer):
        """Should have dispersion_index."""
        result = analyzer.test_clustering()
        assert 'dispersion_index' in result
    
    def test_dispersion_non_negative(self, analyzer):
        """Dispersion index should be non-negative."""
        result = analyzer.test_clustering()
        if result['dispersion_index'] is not None:
            assert result['dispersion_index'] >= 0


class TestInterArrivalTimes:
    """Tests for inter-arrival time methods."""
    
    def test_calculate_returns_series(self, analyzer):
        """calculate_inter_arrival_times should return Series."""
        result = analyzer.calculate_inter_arrival_times()
        assert isinstance(result, pd.Series)
    
    def test_statistics_returns_dict(self, analyzer):
        """get_inter_arrival_statistics should return dict."""
        result = analyzer.get_inter_arrival_statistics()
        assert isinstance(result, dict)
    
    def test_statistics_has_mean(self, analyzer):
        """Statistics should include mean."""
        result = analyzer.get_inter_arrival_statistics()
        assert 'mean' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
