"""
Spatial and network analysis utilities for PMU disturbance analysis.
Includes geographic clustering, spatial statistics, and network topology analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance_matrix
from scipy.stats import chi2
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def validate_coordinates(df: pd.DataFrame,
                        lat_col: str = 'Latitude',
                        lon_col: str = 'Longitude') -> Dict:
    """
    Validate coordinate data quality.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with coordinates
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name

    Returns:
    --------
    Dict
        Validation results
    """
    results = {
        'total_records': len(df),
        'missing_lat': df[lat_col].isna().sum() if lat_col in df.columns else len(df),
        'missing_lon': df[lon_col].isna().sum() if lon_col in df.columns else len(df),
        'valid_lat_range': 0,
        'valid_lon_range': 0,
        'valid_coordinates': 0
    }

    if lat_col in df.columns and lon_col in df.columns:
        # Check valid ranges (lat: -90 to 90, lon: -180 to 180)
        valid_lat = df[lat_col].between(-90, 90)
        valid_lon = df[lon_col].between(-180, 180)

        results['valid_lat_range'] = valid_lat.sum()
        results['valid_lon_range'] = valid_lon.sum()
        results['valid_coordinates'] = (valid_lat & valid_lon & df[lat_col].notna() & df[lon_col].notna()).sum()

    return results


def perform_dbscan_clustering(df: pd.DataFrame,
                              lat_col: str = 'Latitude',
                              lon_col: str = 'Longitude',
                              eps: float = 0.5,
                              min_samples: int = 5) -> pd.DataFrame:
    """
    Perform DBSCAN clustering on geographic coordinates.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with coordinates
    lat_col : str
        Latitude column
    lon_col : str
        Longitude column
    eps : float
        Maximum distance between points in a cluster
    min_samples : int
        Minimum samples in a neighborhood

    Returns:
    --------
    pd.DataFrame
        Original data with cluster assignments
    """
    df = df.copy()

    # Filter valid coordinates
    valid_coords = df[[lat_col, lon_col]].dropna()

    if len(valid_coords) == 0:
        df['Cluster'] = -1
        return df

    # Perform DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clustering.fit_predict(valid_coords)

    # Assign clusters
    df.loc[valid_coords.index, 'Cluster'] = clusters

    return df


def perform_kmeans_clustering(df: pd.DataFrame,
                              lat_col: str = 'Latitude',
                              lon_col: str = 'Longitude',
                              n_clusters: int = 10) -> pd.DataFrame:
    """
    Perform K-means clustering on geographic coordinates.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with coordinates
    lat_col : str
        Latitude column
    lon_col : str
        Longitude column
    n_clusters : int
        Number of clusters

    Returns:
    --------
    pd.DataFrame
        Original data with cluster assignments
    """
    df = df.copy()

    # Filter valid coordinates
    valid_coords = df[[lat_col, lon_col]].dropna()

    if len(valid_coords) < n_clusters:
        df['Cluster'] = 0
        return df

    # Perform K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(valid_coords)

    # Assign clusters
    df.loc[valid_coords.index, 'Cluster'] = clusters
    df['Cluster_Center_Lat'] = np.nan
    df['Cluster_Center_Lon'] = np.nan

    for i in range(n_clusters):
        mask = df['Cluster'] == i
        if mask.sum() > 0:
            df.loc[mask, 'Cluster_Center_Lat'] = kmeans.cluster_centers_[i, 0]
            df.loc[mask, 'Cluster_Center_Lon'] = kmeans.cluster_centers_[i, 1]

    return df


def calculate_morans_i(df: pd.DataFrame,
                      value_col: str,
                      lat_col: str = 'Latitude',
                      lon_col: str = 'Longitude',
                      threshold_distance: float = 1.0) -> Dict:
    """
    Calculate Moran's I spatial autocorrelation statistic.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with coordinates and values
    value_col : str
        Column containing values to test for spatial correlation
    lat_col : str
        Latitude column
    lon_col : str
        Longitude column
    threshold_distance : float
        Distance threshold for neighbors

    Returns:
    --------
    Dict
        Moran's I statistic and significance
    """
    # Filter valid data
    data = df[[lat_col, lon_col, value_col]].dropna()

    if len(data) < 3:
        return {'error': 'Insufficient data for Morans I calculation'}

    # Calculate distance matrix
    coords = data[[lat_col, lon_col]].values
    dist_matrix = distance_matrix(coords, coords)

    # Create spatial weights matrix (binary: 1 if within threshold, 0 otherwise)
    W = (dist_matrix < threshold_distance) & (dist_matrix > 0)
    W = W.astype(float)

    # Normalize weights
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums[:, np.newaxis]

    # Calculate Moran's I
    values = data[value_col].values
    n = len(values)
    mean_value = values.mean()

    # Numerator
    deviations = values - mean_value
    numerator = np.sum(W * np.outer(deviations, deviations))

    # Denominator
    denominator = np.sum(deviations ** 2)

    if denominator == 0:
        return {'error': 'No variation in values'}

    I = (n / W.sum()) * (numerator / denominator)

    # Expected value and variance (under null hypothesis of no spatial autocorrelation)
    E_I = -1 / (n - 1)

    # Simplified variance calculation
    VAR_I = 1 / (n - 1)

    # Z-score
    z_score = (I - E_I) / np.sqrt(VAR_I)

    # P-value (two-tailed test)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return {
        'Morans_I': I,
        'Expected_I': E_I,
        'Z_score': z_score,
        'P_value': p_value,
        'Significant': p_value < 0.05,
        'Interpretation': 'Positive spatial autocorrelation' if I > E_I and p_value < 0.05 else
                         'Negative spatial autocorrelation' if I < E_I and p_value < 0.05 else
                         'No significant spatial autocorrelation'
    }


def build_proximity_network(df: pd.DataFrame,
                            lat_col: str = 'Latitude',
                            lon_col: str = 'Longitude',
                            id_col: str = 'SectionID',
                            threshold_distance: float = 1.0) -> nx.Graph:
    """
    Build network graph based on geographic proximity.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with coordinates
    lat_col : str
        Latitude column
    lon_col : str
        Longitude column
    id_col : str
        Node identifier column
    threshold_distance : float
        Distance threshold for creating edges

    Returns:
    --------
    nx.Graph
        Network graph
    """
    # Filter valid coordinates
    data = df[[id_col, lat_col, lon_col]].dropna()

    # Create graph
    G = nx.Graph()

    # Add nodes
    for idx, row in data.iterrows():
        G.add_node(row[id_col], lat=row[lat_col], lon=row[lon_col])

    # Calculate distances and add edges
    coords = data[[lat_col, lon_col]].values
    dist_matrix = distance_matrix(coords, coords)

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if dist_matrix[i, j] < threshold_distance:
                node_i = data.iloc[i][id_col]
                node_j = data.iloc[j][id_col]
                G.add_edge(node_i, node_j, weight=dist_matrix[i, j])

    return G


def calculate_network_centrality(G: nx.Graph) -> pd.DataFrame:
    """
    Calculate network centrality metrics.

    Parameters:
    -----------
    G : nx.Graph
        Network graph

    Returns:
    --------
    pd.DataFrame
        Centrality metrics for each node
    """
    if len(G.nodes()) == 0:
        return pd.DataFrame()

    metrics = {
        'Node': list(G.nodes()),
        'Degree': [G.degree(node) for node in G.nodes()],
    }

    # Calculate centrality metrics if network is connected enough
    if nx.is_connected(G):
        metrics['Betweenness'] = list(nx.betweenness_centrality(G).values())
        metrics['Closeness'] = list(nx.closeness_centrality(G).values())
        metrics['Eigenvector'] = list(nx.eigenvector_centrality(G, max_iter=1000).values())
    else:
        # Calculate for largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)

        betweenness = nx.betweenness_centrality(subG)
        closeness = nx.closeness_centrality(subG)
        eigenvector = nx.eigenvector_centrality(subG, max_iter=1000)

        metrics['Betweenness'] = [betweenness.get(node, 0) for node in G.nodes()]
        metrics['Closeness'] = [closeness.get(node, 0) for node in G.nodes()]
        metrics['Eigenvector'] = [eigenvector.get(node, 0) for node in G.nodes()]

    df = pd.DataFrame(metrics)
    return df.sort_values('Betweenness', ascending=False)


def detect_communities(G: nx.Graph) -> Dict:
    """
    Detect communities using Louvain algorithm.

    Parameters:
    -----------
    G : nx.Graph
        Network graph

    Returns:
    --------
    Dict
        Community assignments
    """
    try:
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(G)

        # Create node-to-community mapping
        node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community[node] = i

        return {
            'node_community': node_community,
            'num_communities': len(communities),
            'modularity': community.modularity(G, communities)
        }
    except Exception as e:
        return {'error': str(e)}


def test_voltage_level_association(df: pd.DataFrame,
                                   voltage_col: str = 'Voltage',
                                   has_disturbance_col: Optional[str] = None) -> Dict:
    """
    Test association between voltage level and disturbance occurrence using chi-square test.

    Parameters:
    -----------
    df : pd.DataFrame
        PMU data with voltage levels
    voltage_col : str
        Voltage column
    has_disturbance_col : str, optional
        Binary column indicating disturbance presence

    Returns:
    --------
    Dict
        Chi-square test results
    """
    from scipy.stats import chi2_contingency

    if has_disturbance_col is None:
        return {'error': 'Disturbance indicator column required'}

    # Create contingency table
    contingency_table = pd.crosstab(df[voltage_col], df[has_disturbance_col])

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05,
        'contingency_table': contingency_table,
        'interpretation': f'Voltage level and disturbances are {"" if p_value < 0.05 else "not "}significantly associated (p={p_value:.4f})'
    }
