"""
Configuration file for Section 150 deep-dive analysis.
Extends main EDA config with Section 150-specific settings.
"""

import os
import sys

# Add parent and EDA directories to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
EDA_DIR = os.path.join(PARENT_DIR, 'EDA')
sys.path.insert(0, EDA_DIR)
sys.path.insert(0, os.path.join(EDA_DIR, 'src'))

# Import base config from EDA
from config import (
    EXCEL_FILE, PMU_SHEET, DISTURBANCE_SHEET,
    CONFIDENCE_LEVEL, ALPHA, RANDOM_SEED,
    FIGURE_DPI, DEFAULT_FIGSIZE, COLOR_PALETTE, PLOT_SETTINGS
)

# Section 150 specific settings
TARGET_SECTION_ID = 150
N_SIMILAR_SECTIONS = 10

# Output directories for Section 150
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')

# Cached data paths
CACHED_SECTION150_DATA = os.path.join(DATA_DIR, 'section150_events.parquet')
CACHED_NETWORK_STATS = os.path.join(DATA_DIR, 'network_baselines.parquet')
CACHED_SIMILAR_SECTIONS = os.path.join(DATA_DIR, 'similar_sections.parquet')

# Analysis parameters
TEMPORAL_CLUSTERING_WINDOW_DAYS = 7  # Window for burst detection
BURST_THRESHOLD_ZSCORE = 2.0  # Z-score threshold for burst detection

# Risk weights (inherited from EDA, can override if needed)
CAUSE_SEVERITY_WEIGHTS = {
    'Weather': 1.0,
    'Equipment Failure': 0.9,
    'Lightning': 0.8,
    'Wildlife': 0.7,
    'Vegetation': 0.6,
    'Human Error': 0.5,
    'Unknown': 0.4,
    'Other': 0.3
}

# Visualization settings for Section 150 analysis
SECTION150_FIGSIZE = (14, 8)
COMPARISON_FIGSIZE = (16, 10)
HEATMAP_FIGSIZE = (12, 8)

# Report settings
EXECUTIVE_SUMMARY_FILE = os.path.join(REPORT_DIR, 'section150_executive_summary.md')
TECHNICAL_REPORT_FILE = os.path.join(REPORT_DIR, 'section150_technical_report.md')

# Unknown analysis parameters
UNKNOWN_TIME_LAG_HOURS = 48  # Window for finding related events (±48h)
UNKNOWN_MIN_CLUE_CONFIDENCE = 0.7  # Threshold for reclassification

# Operational context patterns
CROMWELL_TAP_PATTERNS = [
    r'Cromwell Tap 31 open',
    r'Cromwell Tap 31 closed',
    r'Cromwell 31 open',
    r'Cromwell.*closed'
]

INTERCONNECTION_KEYWORDS = {
    'PSO': 1.0,      # Weight for criticality score
    'OMPA': 0.8,
    'DCC': 0.6,
    'Weleetka': 0.5,
    'Wetumka': 0.5
}

# Hourly analysis
PEAK_HOUR_FOCUS = 19  # 7 PM

# Visualization file names
FIGURE_NAMES = {
    'timeline': 'fig1_section150_event_timeline',
    'cause_distribution': 'fig2_section150_cause_distribution',
    'interarrival': 'fig3_section150_interarrival_analysis',
    'cyclical_patterns': 'fig4_section150_cyclical_patterns',
    'similar_sections': 'fig5_section150_vs_similar_sections',
    'pmu_characteristics': 'fig6_section150_pmu_characteristics',
    'cumulative_events': 'fig7_section150_cumulative_events',
    # Extended analysis figures
    'cromwell_analysis': 'fig8_cromwell_tap_operational_state',
    'unknown_reclassification': 'fig9_unknown_events_reclassification',
    'hourly_comparison': 'fig10_hourly_distribution_comparison',
    '7pm_deep_dive': 'fig11_7pm_peak_multifactor_analysis',
    'day_hour_heatmap': 'fig12_day_hour_event_heatmap'
}
