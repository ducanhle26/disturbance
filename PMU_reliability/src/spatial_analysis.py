"""
Spatial Analysis Module for PMU Reliability Framework.

Geographic clustering and similarity analysis for PMU sections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class SpatialAnalyzer:
    """
    Spatial and similarity analyzer for PMU sections.
    
    Provides methods for:
    - Geographic clustering using DBSCAN
    - Finding similar sections based on characteristics
    - Disturbance density analysis
    """
    
    def __init__(self, pmu_df: pd.DataFrame, dist_df: pd.DataFrame):
        """
        Initialize spatial analyzer.
        
        Parameters
        ----------
        pmu_df : pd.DataFrame
            PMU installations dataframe
        dist_df : pd.DataFrame
            Disturbance events dataframe
        """
        self.pmu_df = pmu_df.copy()
        self.dist_df = dist_df.copy()
        
        self.lat_col = self._find_column(['lat', 'latitude'])
        self.lon_col = self._find_column(['lon', 'long', 'longitude'])
        self.voltage_col = self._find_column(['volt', 'kv', 'voltage'])
        self.type_col = self._find_column(['type', 'pmu_type', 'pmutype'])
        
        self._add_event_counts()
    
    def _find_column(self, keywords: List[str]) -> Optional[str]:
        """Find column matching any of the keywords."""
        for col in self.pmu_df.columns:
            for kw in keywords:
                if kw in col.lower():
                    return col
        return None
    
    def _add_event_counts(self):
        """Add event counts to PMU dataframe."""
        events_per_section = self.dist_df.groupby('SectionID').size()
        self.pmu_df['Event_Count'] = (
            self.pmu_df['SectionID'].map(events_per_section).fillna(0).astype(int)
        )
    
    def cluster_geographic(self, eps: float = 0.5, 
                           min_samples: int = 3) -> pd.DataFrame:
        """
        Cluster PMU sections geographically using DBSCAN.
        
        Parameters
        ----------
        eps : float
            Maximum distance between points in a cluster (in degrees, ~55km per degree)
        min_samples : int
            Minimum points to form a cluster
            
        Returns
        -------
        pd.DataFrame
            PMU dataframe with cluster_id column
        """
        if self.lat_col is None or self.lon_col is None:
            raise ValueError("Geographic coordinates not found in PMU data")
        
        coords = self.pmu_df[[self.lat_col, self.lon_col]].dropna()
        
        if len(coords) < min_samples:
            self.pmu_df['cluster_id'] = -1
            return self.pmu_df
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        self.pmu_df.loc[coords.index, 'cluster_id'] = clustering.fit_predict(coords)
        self.pmu_df['cluster_id'] = self.pmu_df['cluster_id'].fillna(-1).astype(int)
        
        return self.pmu_df
    
    def find_similar_sections(self, section_id: int, k: int = 10,
                              use_event_count: bool = False) -> pd.DataFrame:
        """
        Find sections most similar to a given section.
        
        Similarity is based on voltage level, PMU type, age, and optionally location.
        Event count is excluded by default to avoid data leakage.
        
        Parameters
        ----------
        section_id : int
            Target section ID
        k : int
            Number of similar sections to return
        use_event_count : bool
            Include event count in similarity (default False to avoid leakage)
            
        Returns
        -------
        pd.DataFrame
            Most similar sections with similarity scores
        """
        target = self.pmu_df[self.pmu_df['SectionID'] == section_id]
        if len(target) == 0:
            raise ValueError(f"Section {section_id} not found")
        
        feature_cols = []
        
        if self.voltage_col:
            feature_cols.append(self.voltage_col)
        
        if 'Age_Years' in self.pmu_df.columns:
            feature_cols.append('Age_Years')
        
        if self.lat_col and self.lon_col:
            feature_cols.extend([self.lat_col, self.lon_col])
        
        if use_event_count:
            feature_cols.append('Event_Count')
        
        if not feature_cols:
            return self._find_similar_by_category(section_id, k)
        
        features = self.pmu_df[['SectionID'] + feature_cols].copy()
        features = features.dropna()
        
        if section_id not in features['SectionID'].values:
            return self._find_similar_by_category(section_id, k)
        
        feature_matrix = features[feature_cols].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        target_idx = features[features['SectionID'] == section_id].index[0]
        target_features = scaled_features[features.index.get_loc(target_idx)].reshape(1, -1)
        
        similarities = cosine_similarity(target_features, scaled_features)[0]
        
        features['similarity'] = similarities
        
        similar = features[features['SectionID'] != section_id].copy()
        similar = similar.sort_values('similarity', ascending=False)
        
        result = similar.head(k).merge(
            self.pmu_df[['SectionID', 'Event_Count'] + 
                        ([self.voltage_col] if self.voltage_col else []) +
                        ([self.type_col] if self.type_col else [])],
            on='SectionID',
            how='left',
            suffixes=('', '_dup')
        )
        
        dup_cols = [c for c in result.columns if c.endswith('_dup')]
        result = result.drop(columns=dup_cols)
        
        return result
    
    def _find_similar_by_category(self, section_id: int, k: int) -> pd.DataFrame:
        """Find similar sections by categorical matching (voltage, type)."""
        target = self.pmu_df[self.pmu_df['SectionID'] == section_id].iloc[0]
        
        mask = self.pmu_df['SectionID'] != section_id
        
        if self.voltage_col and pd.notna(target.get(self.voltage_col)):
            mask &= self.pmu_df[self.voltage_col] == target[self.voltage_col]
        
        if self.type_col and pd.notna(target.get(self.type_col)):
            mask &= self.pmu_df[self.type_col] == target[self.type_col]
        
        similar = self.pmu_df[mask].copy()
        similar['similarity'] = 1.0  # Perfect categorical match
        similar = similar.sort_values('Event_Count', ascending=True)
        
        return similar.head(k)
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each geographic cluster.
        
        Returns
        -------
        pd.DataFrame
            Cluster summary with counts, event statistics
        """
        if 'cluster_id' not in self.pmu_df.columns:
            self.cluster_geographic()
        
        summary = self.pmu_df.groupby('cluster_id').agg({
            'SectionID': 'count',
            'Event_Count': ['sum', 'mean', 'max']
        }).round(2)
        
        summary.columns = ['num_sections', 'total_events', 'mean_events', 'max_events']
        summary = summary.reset_index()
        summary = summary.sort_values('total_events', ascending=False)
        
        return summary
    
    def calculate_disturbance_density(self, grid_size: float = 1.0) -> pd.DataFrame:
        """
        Calculate disturbance density on a geographic grid.
        
        Parameters
        ----------
        grid_size : float
            Grid cell size in degrees
            
        Returns
        -------
        pd.DataFrame
            Grid cells with disturbance densities
        """
        if self.lat_col is None or self.lon_col is None:
            raise ValueError("Geographic coordinates not found")
        
        merged = self.dist_df.merge(
            self.pmu_df[['SectionID', self.lat_col, self.lon_col]],
            on='SectionID',
            how='left'
        )
        
        merged = merged.dropna(subset=[self.lat_col, self.lon_col])
        
        merged['lat_grid'] = (merged[self.lat_col] // grid_size) * grid_size
        merged['lon_grid'] = (merged[self.lon_col] // grid_size) * grid_size
        
        density = merged.groupby(['lat_grid', 'lon_grid']).size().reset_index(name='event_count')
        density['density'] = density['event_count'] / (grid_size ** 2)
        
        return density.sort_values('density', ascending=False)
