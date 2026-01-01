# PMU Disturbance Analysis - Notebooks Guide

## All 7 Analysis Notebooks Created! ✅

### Notebook Overview

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| **01_data_loading_quality.ipynb** | Data loading & QA | Data quality report, cleaned dataset |
| **02_temporal_analysis.ipynb** | Time series analysis | Decomposition, anomalies, forecasts |
| **03_causality_patterns.ipynb** | Pattern mining | Cause analysis, association rules, MTBF |
| **04_spatial_network.ipynb** | Geographic analysis | Clustering, spatial maps, network topology |
| **05_predictive_modeling.ipynb** | Risk scoring & forecasting | Risk scores for 533 PMUs, predictions |
| **06_operational_insights.ipynb** | Maintenance analysis | Age analysis, bathtub curve, type comparison |
| **07_statistical_validation.ipynb** | Hypothesis testing | Distribution tests, correlations, ANOVA |

---

## Execution Order

### Sequential Workflow (Recommended)

```
1. Start Here: Notebook 01 (Data Loading)
   ↓
2. Run in Order: Notebooks 02-07
   ↓
3. Review Outputs in: outputs/ directory
```

**Why Sequential?**
- Notebook 01 creates `cleaned_data.parquet` used by all others
- Each notebook saves intermediate results
- Progressive analysis builds on previous findings

### Parallel Workflow (Advanced)

After running Notebook 01, notebooks 02-07 can run **independently** if needed.

---

## Quick Start Guide

### Step 1: Install Dependencies

```bash
cd /Users/anhle/disturbance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Launch Jupyter

```bash
jupyter notebook
```

### Step 3: Run Notebooks

1. **Start with Notebook 01**
   - Navigate to `notebooks/01_data_loading_quality.ipynb`
   - Click "Kernel" → "Restart & Run All"
   - Review the data quality report
   - Verify `outputs/data/cleaned_data.parquet` is created

2. **Continue with Notebooks 02-07**
   - Run each notebook in sequence
   - Review visualizations and results
   - Check `outputs/` directories for saved files

---

## Detailed Notebook Descriptions

### 01_data_loading_quality.ipynb

**Purpose**: Load and validate PMU disturbance data

**What it does**:
- Loads Excel data (PMUs and Disturbances sheets)
- Performs comprehensive data quality checks:
  - Missing values analysis
  - Duplicate detection
  - Outlier identification (IQR method)
  - Temporal coverage validation
  - SectionID linkage verification
- Generates data quality report
- Saves cleaned data for downstream analysis

**Expected Runtime**: 2-5 minutes

**Key Outputs**:
- `outputs/data/cleaned_data.parquet`
- `outputs/data/pmu_data.csv`
- `outputs/data/disturbance_data.csv`
- `outputs/reports/data_quality_report.txt`
- `outputs/figures/static/01_missing_values.png`

---

### 02_temporal_analysis.ipynb

**Purpose**: Analyze disturbance patterns over time

**What it does**:
- **STL Decomposition**: Separates trend, seasonal, and residual components
- **Anomaly Detection**: Identifies unusual spikes using 3 methods (Z-score, IQR, Isolation Forest)
- **Inter-Arrival Times**: Calculates time between disturbances, tests for Poisson process
- **Change Point Detection**: Finds regime shifts using PELT algorithm
- **Cyclical Patterns**: Hour-of-day, day-of-week, monthly analysis
- **Rolling Statistics**: 7/30/90-day moving averages
- **ACF/PACF Analysis**: Identifies temporal correlations

**Expected Runtime**: 5-10 minutes

**Key Outputs**:
- `outputs/data/temporal_results.csv`
- 9 visualizations (both static and interactive):
  - Time series decomposition
  - Anomaly detection plots
  - Inter-arrival time distributions
  - Change point visualization
  - Cyclical pattern charts
  - Rolling statistics
  - Calendar heatmap
  - ACF/PACF plots

---

### 03_causality_patterns.ipynb

**Purpose**: Analyze disturbance causes and relationships

**What it does**:
- **Cause Distribution**: Frequency analysis and Pareto chart (80/20 rule)
- **Cause Evolution**: Trends over time for top causes
- **Association Rules**: Apriori algorithm to find cause co-occurrences
- **Co-occurrence Matrix**: Which causes appear together
- **Sequential Patterns**: Cause A → Cause B within time windows
- **Transition Matrices**: Probability of cause transitions
- **Reliability Metrics**: MTBF, MTTR, failure rates by section
- **Cause Severity**: Impact assessment

**Expected Runtime**: 5-10 minutes

**Key Outputs**:
- `outputs/data/causality_results.csv`
- `outputs/data/reliability_metrics.csv`
- `outputs/data/sequential_patterns.csv`
- 6 visualizations:
  - Pareto chart
  - Cause evolution over time
  - Co-occurrence matrix heatmap
  - Transition probability matrix
  - MTBF analysis plots
  - Sankey diagram (interactive)

---

### 04_spatial_network.ipynb

**Purpose**: Geographic clustering and network topology

**What it does**:
- **Coordinate Validation**: Verifies lat/lon data quality
- **DBSCAN Clustering**: Density-based geographic clustering
- **Interactive Map**: Folium map with PMU locations and clusters
- **Moran's I**: Spatial autocorrelation testing
- **Network Construction**: Builds proximity-based network
- **Centrality Analysis**: Betweenness, closeness, eigenvector metrics
- **Community Detection**: Identifies subnetworks

**Expected Runtime**: 3-5 minutes

**Key Outputs**:
- `outputs/data/spatial_results.csv`
- Interactive map: `outputs/figures/interactive/04_01_pmu_locations_map.html`
- Network centrality analysis

---

### 05_predictive_modeling.ipynb

**Purpose**: Risk scoring and forecasting

**What it does**:
- **Risk Scoring**: Calculates composite risk scores for all 533 PMU sections
  - Uses weighted components (frequency, trend, severity, age, time since last)
  - Assigns risk categories (Low, Medium, High, Critical)
- **ARIMA Forecasting**: 30/60/90-day disturbance predictions
- **Section-Level Predictions**: Expected disturbances per section
- **Forecast Evaluation**: MAE, RMSE, MAPE metrics

**Expected Runtime**: 5-8 minutes

**Key Outputs**:
- `outputs/models/risk_scores_all_sections.csv` ⭐ **Critical Output**
- `outputs/models/predictions_30_60_90_days.csv` ⭐ **Critical Output**
- Visualizations:
  - Risk score distribution
  - ARIMA forecast with confidence intervals

**Most Important Notebook for Actionable Insights!**

---

### 06_operational_insights.ipynb

**Purpose**: PMU age, maintenance, and operational analysis

**What it does**:
- **PMU Age Analysis**: Calculates age from InService dates
- **Age vs Disturbances**: Correlation analysis
- **Bathtub Curve**: Failure rate by age group (infant mortality, useful life, wear-out)
- **PMU Type Comparison**: Reliability differences across types (ANOVA)
- **Service Pattern Analysis**: OutService durations and patterns

**Expected Runtime**: 3-5 minutes

**Key Outputs**:
- `outputs/data/operational_results.csv`
- Visualizations:
  - Age vs disturbances scatter plot
  - Bathtub curve
  - Type comparison box plots

---

### 07_statistical_validation.ipynb

**Purpose**: Hypothesis testing and statistical rigor

**What it does**:
- **Distribution Fitting**: Tests for Poisson, Negative Binomial, Normal distributions
- **Mann-Kendall Test**: Trend significance testing
- **Correlation Analysis**: Spearman/Pearson with significance testing
- **ANOVA**: Voltage level vs disturbance rate relationship
- **Bootstrap CIs**: 95% confidence intervals for key metrics
- **Chi-Square Tests**: Independence between categorical variables
- **Q-Q Plots**: Visual distribution validation

**Expected Runtime**: 3-5 minutes

**Key Outputs**:
- `outputs/data/statistical_validation.csv`
- Visualizations:
  - Q-Q plots
  - Correlation matrix heatmap
  - Forest plots

---

## Expected Outputs Summary

### Files Created (After Running All Notebooks)

```
outputs/
├── data/
│   ├── cleaned_data.parquet
│   ├── pmu_data.csv
│   ├── disturbance_data.csv
│   ├── temporal_results.csv
│   ├── causality_results.csv
│   ├── reliability_metrics.csv
│   ├── sequential_patterns.csv
│   ├── spatial_results.csv
│   ├── operational_results.csv
│   └── statistical_validation.csv
├── models/
│   ├── risk_scores_all_sections.csv  ⭐ KEY OUTPUT
│   └── predictions_30_60_90_days.csv ⭐ KEY OUTPUT
├── reports/
│   └── data_quality_report.txt
└── figures/
    ├── static/        (18-22 PNG/PDF files at 300 DPI)
    └── interactive/   (5-8 HTML files)
```

### Visualization Count

- **Static plots**: 18-22 high-resolution images (PNG + PDF)
- **Interactive plots**: 5-8 HTML files with Plotly/Folium
- **Total**: ~25-30 visualizations

---

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError"**
```bash
pip install [missing_package]
```

**2. "No datetime column found"**
- Check your Excel file column names
- Manually specify datetime column in notebook cells

**3. "No valid coordinates for mapping"**
- Verify PMU data has Latitude/Longitude columns
- Check coordinate format (should be decimal degrees)

**4. "Association rules not found"**
- Lower `MIN_SUPPORT` or `MIN_CONFIDENCE` in `config.py`
- This is normal if causes don't co-occur frequently

**5. Notebook crashes or freezes**
- Restart kernel: "Kernel" → "Restart & Clear Output"
- Check memory usage (9,369 events should fit in RAM)

---

## Customization Tips

### Adjust Analysis Parameters

Edit `config.py` to customize:

```python
# Temporal analysis
ROLLING_WINDOWS = [7, 30, 90]        # Days
FORECAST_HORIZONS = [30, 60, 90]     # Days

# Pattern mining
MIN_SUPPORT = 0.01                   # Lower for more rules
MIN_CONFIDENCE = 0.5                 # Lower for more rules
SEQUENTIAL_WINDOW_DAYS = 7           # Time window for patterns

# Spatial analysis
DBSCAN_EPS = 0.5                     # Clustering distance
DBSCAN_MIN_SAMPLES = 5               # Minimum cluster size

# Risk scoring weights (must sum to 1.0)
RISK_WEIGHTS = {
    'historical_frequency': 0.40,
    'trend_direction': 0.20,
    'cause_severity': 0.20,
    'time_since_last': 0.10,
    'pmu_age': 0.10
}
```

### Modify Visualizations

Edit `config.py`:

```python
FIGURE_DPI = 600                     # Higher resolution
DEFAULT_FIGSIZE = (16, 10)           # Larger figures
COLOR_PALETTE = 'plasma'             # Different colors
```

---

## Next Steps After Running Notebooks

### 1. Review Key Outputs

**Priority 1**: Risk Scores
```bash
head outputs/models/risk_scores_all_sections.csv
```

**Priority 2**: Predictions
```bash
head outputs/models/predictions_30_60_90_days.csv
```

### 2. Create Reports

Use the results to generate:
- Executive summary with top 10 insights
- Maintenance priority list (top 100 highest risk sections)
- Technical report documenting methodology

### 3. Share Results

**For stakeholders**:
- Interactive HTML maps (`outputs/figures/interactive/`)
- Risk score spreadsheet (`risk_scores_all_sections.csv`)
- Summary visualizations (`outputs/figures/static/`)

**For technical teams**:
- Detailed analysis results (all CSV files)
- Methodology documentation (notebook cells)
- Statistical validation results

### 4. Iterate and Refine

Based on initial results:
- Adjust risk weight parameters
- Refine clustering parameters
- Focus on specific high-risk sections
- Deep-dive into specific causes

---

## Performance Expectations

### Total Runtime
- **All 7 notebooks**: ~30-45 minutes (sequential)
- **Parallel (02-07)**: ~10-15 minutes (after notebook 01)

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: ~500MB for outputs
- **Python**: 3.9+ recommended

---

## Summary

You now have:
- ✅ **7 Complete Analysis Notebooks**
- ✅ **All Required Source Modules**
- ✅ **Comprehensive Documentation**
- ✅ **Production-Ready Code**

**Ready to execute the complete PMU disturbance analysis!**

Start with Notebook 01 and work through systematically, or jump to Notebook 05 for immediate risk scoring results.

---

*For detailed module documentation, see source code docstrings in `src/`*
*For configuration options, see `config.py`*
*For overall project guidance, see `README.md` and `EXECUTION_SUMMARY.md`*
