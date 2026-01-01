"""
Configuration file for PMU Disturbance Analysis project.
Contains file paths, analysis parameters, and output settings.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')

# Input data paths
EXCEL_FILE = os.path.join(DATA_DIR, 'PMU_disturbance.xlsx')
PMU_SHEET = 'PMUs'
DISTURBANCE_SHEET = 'Disturbances'

# Output data paths
CLEANED_DATA = os.path.join(OUTPUT_DIR, 'data', 'cleaned_data.parquet')
TEMPORAL_RESULTS = os.path.join(OUTPUT_DIR, 'data', 'temporal_results.csv')
CAUSALITY_RESULTS = os.path.join(OUTPUT_DIR, 'data', 'causality_results.csv')
SPATIAL_RESULTS = os.path.join(OUTPUT_DIR, 'data', 'spatial_results.csv')
OPERATIONAL_RESULTS = os.path.join(OUTPUT_DIR, 'data', 'operational_results.csv')
STATISTICAL_RESULTS = os.path.join(OUTPUT_DIR, 'data', 'statistical_validation.csv')

# Model output paths
RISK_SCORES = os.path.join(MODEL_DIR, 'risk_scores_all_sections.csv')
PREDICTIONS = os.path.join(MODEL_DIR, 'predictions_30_60_90_days.csv')

# Statistical parameters
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05  # Significance level
RANDOM_SEED = 42

# Time series parameters
ROLLING_WINDOWS = [7, 30, 90]  # Days
FORECAST_HORIZONS = [30, 60, 90]  # Days

# Pattern mining parameters
MIN_SUPPORT = 0.01  # Minimum support for association rules
MIN_CONFIDENCE = 0.5  # Minimum confidence for association rules
SEQUENTIAL_WINDOW_DAYS = 7  # Time window for sequential patterns

# Spatial analysis parameters
DBSCAN_EPS = 0.5  # DBSCAN epsilon (adjust based on coordinate scale)
DBSCAN_MIN_SAMPLES = 5
KMEANS_N_CLUSTERS = 10

# Risk scoring weights
RISK_WEIGHTS = {
    'historical_frequency': 0.40,
    'trend_direction': 0.20,
    'cause_severity': 0.20,
    'time_since_last': 0.10,
    'pmu_age': 0.10
}

# Visualization settings
FIGURE_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']  # Static formats
INTERACTIVE_FORMAT = 'html'
DEFAULT_FIGSIZE = (12, 8)
COLOR_PALETTE = 'viridis'  # Colorblind-friendly
MAP_TILE_LAYER = 'OpenStreetMap'  # Options: 'OpenStreetMap', 'CartoDB positron', 'Stamen Terrain'

# Plot-specific settings
PLOT_SETTINGS = {
    'font_size': 12,
    'title_size': 14,
    'label_size': 12,
    'legend_size': 10,
    'style': 'whitegrid'  # seaborn style
}
