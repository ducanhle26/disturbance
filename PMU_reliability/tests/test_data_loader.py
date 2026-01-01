"""
Unit Tests for Data Loader Module.
"""

import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import (
    load_pmu_disturbance_data, 
    get_section_events, 
    calculate_event_statistics,
    get_network_statistics
)

DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'PMU_disturbance.xlsx'


@pytest.fixture(scope='module')
def data():
    """Load data once for all tests."""
    return load_pmu_disturbance_data(str(DATA_FILE))


class TestLoadData:
    """Tests for load_pmu_disturbance_data function."""
    
    def test_returns_tuple(self, data):
        """Should return tuple of two dataframes."""
        assert isinstance(data, tuple)
        assert len(data) == 2
    
    def test_dataframes_not_empty(self, data):
        """Both dataframes should have data."""
        pmu_df, dist_df = data
        assert len(pmu_df) > 0
        assert len(dist_df) > 0
    
    def test_section_id_column_exists(self, data):
        """SectionID should exist in both dataframes."""
        pmu_df, dist_df = data
        assert 'SectionID' in pmu_df.columns
        assert 'SectionID' in dist_df.columns
    
    def test_date_columns_parsed(self, data):
        """Date columns should be datetime type."""
        pmu_df, _ = data
        if 'InService' in pmu_df.columns:
            assert pmu_df['InService'].dtype == 'datetime64[ns]'
    
    def test_age_columns_added(self, data):
        """Age columns should be added if InService exists."""
        pmu_df, _ = data
        if 'InService' in pmu_df.columns:
            assert 'Age_Days' in pmu_df.columns
            assert 'Age_Years' in pmu_df.columns


class TestGetSectionEvents:
    """Tests for get_section_events function."""
    
    def test_returns_dataframe(self, data):
        """Should return a DataFrame."""
        _, dist_df = data
        result = get_section_events(dist_df, 150)
        assert isinstance(result, pd.DataFrame)
    
    def test_filters_correctly(self, data):
        """All returned events should be for specified section."""
        _, dist_df = data
        result = get_section_events(dist_df, 150)
        assert all(result['SectionID'] == 150)
    
    def test_empty_for_nonexistent_section(self, data):
        """Should return empty DataFrame for nonexistent section."""
        _, dist_df = data
        result = get_section_events(dist_df, 999999)
        assert len(result) == 0


class TestCalculateEventStatistics:
    """Tests for calculate_event_statistics function."""
    
    def test_returns_dict(self, data):
        """Should return a dictionary."""
        _, dist_df = data
        events = get_section_events(dist_df, 150)
        result = calculate_event_statistics(events)
        assert isinstance(result, dict)
    
    def test_has_required_keys(self, data):
        """Should have all required keys."""
        _, dist_df = data
        events = get_section_events(dist_df, 150)
        result = calculate_event_statistics(events)
        
        required_keys = ['count', 'mtbf_days', 'first_event', 'last_event', 'span_days']
        for key in required_keys:
            assert key in result
    
    def test_count_is_correct(self, data):
        """Count should match actual event count."""
        _, dist_df = data
        events = get_section_events(dist_df, 150)
        result = calculate_event_statistics(events)
        assert result['count'] == len(events)
    
    def test_empty_events(self):
        """Should handle empty DataFrame."""
        empty_df = pd.DataFrame(columns=['SectionID', 'DateTime'])
        result = calculate_event_statistics(empty_df)
        assert result['count'] == 0
        assert result['mtbf_days'] is None


class TestNetworkStatistics:
    """Tests for get_network_statistics function."""
    
    def test_returns_dict(self, data):
        """Should return a dictionary."""
        pmu_df, dist_df = data
        result = get_network_statistics(dist_df, pmu_df)
        assert isinstance(result, dict)
    
    def test_has_required_keys(self, data):
        """Should have all required keys."""
        pmu_df, dist_df = data
        result = get_network_statistics(dist_df, pmu_df)
        
        required_keys = ['total_events', 'total_sections', 'mean_events_per_section']
        for key in required_keys:
            assert key in result


class TestEdgeCases:
    """Edge case tests."""
    
    def test_single_event_section(self, data):
        """Should handle sections with only one event."""
        _, dist_df = data
        
        event_counts = dist_df.groupby('SectionID').size()
        single_event_section = event_counts[event_counts == 1].index[0] if len(event_counts[event_counts == 1]) > 0 else None
        
        if single_event_section:
            events = get_section_events(dist_df, single_event_section)
            result = calculate_event_statistics(events)
            assert result['count'] == 1
            assert result['mtbf_days'] is None  # Can't calculate MTBF with 1 event


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
