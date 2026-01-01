# PMU Disturbance Analysis Project

Comprehensive analysis of 533 PMU installations and 9,369 disturbance events from power grid monitoring data.

## Project Overview

This project provides:
- **7 Analysis Categories**: Temporal, causality, spatial, predictive, operational, statistical validation
- **Risk Scoring**: Composite risk scores for all 533 PMU sections
- **Predictive Models**: Time series forecasting and survival analysis
- **15-20 Publication-Quality Visualizations**: Both static (PNG/PDF) and interactive (HTML) formats
- **Comprehensive Reports**: Executive summary, technical documentation, and actionable recommendations

## Installation

### Prerequisites
- Python 3.9+ recommended
- pip package manager

### Setup Instructions

1. **Clone/Navigate to the project directory**:
   ```bash
   cd /Users/anhle/disturbance
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Note**: Some packages may require additional system dependencies:
   - **geopandas**: May need GDAL installed
     ```bash
     # macOS
     brew install gdal

     # Ubuntu/Debian
     sudo apt-get install gdal-bin libgdal-dev
     ```

4. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, statsmodels, networkx, ruptures, lifelines, mlxtend, plotly, folium; print('All packages installed successfully!')"
   ```

## Project Structure

```
disturbance/
├── data/
│   └── PMU_disturbance.xlsx          # Input data (2 sheets)
├── notebooks/
│   ├── 01_data_loading_quality.ipynb     # Data loading & QA
│   ├── 02_temporal_analysis.ipynb        # Time series analysis
│   ├── 03_causality_patterns.ipynb       # Pattern mining
│   ├── 04_spatial_network.ipynb          # Geographic analysis
│   ├── 05_predictive_modeling.ipynb      # Risk scoring & forecasting
│   ├── 06_operational_insights.ipynb     # PMU age & maintenance
│   └── 07_statistical_validation.ipynb   # Hypothesis testing
├── src/
│   ├── data_loader.py                # Data loading utilities
│   ├── temporal.py                   # Time series functions
│   ├── causality.py                  # Pattern mining
│   ├── spatial.py                    # Geographic analysis
│   ├── predictive.py                 # ML & forecasting
│   ├── statistical.py                # Statistical tests
│   └── visualizations.py             # Plotting utilities
├── outputs/
│   ├── figures/
│   │   ├── static/                   # PNG and PDF (300 DPI)
│   │   └── interactive/              # HTML interactive plots
│   ├── models/                       # Risk scores & predictions
│   ├── reports/                      # Analysis reports
│   └── data/                         # Intermediate results
├── config.py                         # Project configuration
├── requirements.txt                  # Python dependencies
├── CLAUDE.md                         # Claude Code documentation
└── README.md                         # This file
```

## Quick Start

### Option 1: Run Notebook 01 First (Recommended)

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open and run** `notebooks/01_data_loading_quality.ipynb`:
   - This notebook loads the data, performs quality checks, and saves cleaned data
   - Review the data quality report generated in `outputs/reports/`

3. **Continue with subsequent notebooks** (02-07) in order:
   - Each notebook builds on the previous one
   - Notebooks can be run independently after Notebook 01 completes

### Option 2: Python Scripts (Advanced)

If you prefer to run analyses as scripts instead of notebooks:

```python
import sys
sys.path.append('/Users/anhle/disturbance')

from src.data_loader import load_all_data
import config

# Load data
pmu_df, disturbance_df, merged_df = load_all_data(
    config.EXCEL_FILE,
    config.PMU_SHEET,
    config.DISTURBANCE_SHEET
)

# Use analysis modules
from src import temporal, causality, spatial, predictive, statistical

# Example: Calculate risk scores
risk_scores = predictive.calculate_composite_risk_score(pmu_df, disturbance_df)
print(risk_scores.head())
```

## Execution Workflow

The analysis follows a **sequential workflow** with dependencies:

```
Phase 1: Data Loading (Notebook 01)
    ↓
Phase 2-7: Analysis Notebooks (can run independently after Phase 1)
    ├── Temporal Analysis (02)
    ├── Causality & Patterns (03)
    ├── Spatial & Network (04)
    ├── Predictive Modeling (05)
    ├── Operational Insights (06)
    └── Statistical Validation (07)
    ↓
Phase 8-9: Report Generation & Visualization Consolidation
```

## Key Configuration

Edit `config.py` to customize:

- **File Paths**: Input/output locations
- **Analysis Parameters**:
  - Confidence levels (default: 95%)
  - Rolling window sizes (7, 30, 90 days)
  - Forecast horizons (30, 60, 90 days)
  - Association rule thresholds (min support, confidence)
- **Risk Scoring Weights**:
  - Historical frequency: 40%
  - Trend direction: 20%
  - Cause severity: 20%
  - Time since last: 10%
  - PMU age: 10%
- **Visualization Settings**:
  - DPI: 300 (publication quality)
  - Color palette: 'viridis' (colorblind-friendly)
  - Figure size: (12, 8)

## Expected Outputs

### Data Products
- `outputs/data/cleaned_data.parquet` - Clean merged dataset
- `outputs/data/*_results.csv` - Results from each analysis category
- `outputs/models/risk_scores_all_sections.csv` - Risk scores for 533 PMUs
- `outputs/models/predictions_30_60_90_days.csv` - Forecasts

### Visualizations
- **Static**: 15-20 plots in PNG (300 DPI) and PDF formats
- **Interactive**: HTML versions with plotly/folium for exploration
- Categories: Temporal, Causality, Spatial, Predictive, Operational, Statistical

### Reports
- `outputs/reports/data_quality_report.txt` - Data QA summary
- `outputs/reports/executive_summary.md` - Top insights & recommendations
- `outputs/reports/technical_report.md` - Detailed methodology
- `outputs/reports/recommendations.md` - Actionable maintenance priorities

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'X'**
   ```bash
   pip install X
   ```

2. **Geopandas installation fails**:
   - Install GDAL first (see Installation section)
   - Or skip geopandas and use alternative mapping: `pip install folium`

3. **Memory issues with large datasets**:
   - The dataset (9,369 events) should fit in memory
   - If issues persist, adjust chunk processing in notebooks

4. **Plotly figures not showing in Jupyter**:
   ```bash
   pip install jupyterlab "ipywidgets>=7.5"
   ```

5. **Coordinate validation errors**:
   - Check if PMU data has 'Latitude' and 'Longitude' columns
   - Verify coordinates are in decimal degrees (EPSG:4326)

## Next Steps

1. **Run Notebook 01** to load data and perform quality assessment
2. **Review data quality report** to understand any data issues
3. **Execute analysis notebooks (02-07)** based on priorities:
   - For quick insights: Start with 05 (Predictive Modeling) for risk scores
   - For comprehensive analysis: Run sequentially 02-07
4. **Generate visualizations** and consolidate outputs
5. **Create final reports** with key findings and recommendations

## Analysis Details

For detailed information about:
- **Analysis requirements**: See `EDA/PMU_disturbance_EDA.md`
- **Implementation plan**: See `/Users/anhle/.claude/plans/velvety-yawning-hoare.md`
- **Claude Code guidance**: See `CLAUDE.md`

## Support

For questions or issues with this analysis framework, refer to:
- Notebook documentation (markdown cells in each notebook)
- Function docstrings in `src/` modules
- Configuration parameters in `config.py`

## License

This project is for internal analysis use. All data should be handled according to your organization's data policies.
