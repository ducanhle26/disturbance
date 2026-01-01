# Section 150 Deep-Dive Analysis

This folder contains a comprehensive investigation of **Section 150**, the highest-risk PMU section in the network with **301 disturbance events** (25x the network average).

## Quick Start

```bash
# Activate virtual environment
source ../venv/bin/activate

# Run complete analysis
python run_section150.py
```

## Project Structure

```
Section_150/
├── config_section150.py           # Configuration settings
├── run_section150.py              # Main analysis runner
├── investigate_section150.md      # Analysis requirements
├── README.md                      # This file
├── src/
│   ├── __init__.py
│   ├── section150_loader.py       # Data loading & network baselines
│   ├── section150_event_breakdown.py   # Timeline, clustering, causes
│   ├── section150_characteristics.py   # PMU metadata & similarity
│   ├── section150_root_cause.py        # Statistical tests & patterns
│   ├── section150_comparative.py       # Similar sections analysis
│   ├── section150_recommendations.py   # Actionable interventions
│   ├── section150_visuals.py           # Visualization generation
│   └── section150_report.py            # Report generation
├── outputs/
│   ├── data/                      # Cached analysis data
│   ├── figures/
│   │   ├── static/                # PNG images (7 figures)
│   │   └── interactive/           # Interactive HTML plots
│   └── reports/
│       ├── section150_executive_summary.md
│       └── section150_technical_report.md
└── notebooks/                     # Jupyter notebooks (optional)
```

## Key Findings

| Metric | Value | Significance |
|--------|-------|--------------|
| Total Events | **301** | Highest in network (#1 of 533) |
| Rate vs Average | **25.1x** | Significantly higher (p<0.001) |
| MTBF | **16.4 days** | 10.4x faster than network average |
| Event Clustering | **Clustered** | Dispersion Index: 1.50 |
| Peak Period | **May, 7PM, Thursday** | Significant seasonal variation |

## Analysis Modules

### 1. Event Breakdown
- Complete timeline of all 301 events
- Temporal clustering analysis (events are bunched, not random)
- Cause distribution with network comparison
- Statistical significance testing

### 2. Section Characteristics
- PMU metadata extraction (voltage, type, age, location)
- Similarity matching with other sections
- Identification of unique features

### 3. Root Cause Analysis
- Poisson rate test: Confirms significantly elevated rate
- Top 5 causes with relative risk calculations
- Inter-arrival time analysis (MTBF: 16.4 days vs 170 days network)
- Seasonal/cyclical pattern detection

### 4. Comparative Analysis
- 23 similar sections identified (same voltage & PMU type)
- Best-performing similar section has only 3 events
- Key learnings extracted for improvement

### 5. Recommendations
- Priority 1: Address weather-related events (lightning, severe weather)
- Priority 2: Increase monitoring during peak periods (May, 7PM)
- Priority 3: Apply best practices from low-event similar sections
- **Expected reduction: 10-30% with targeted interventions**

## Generated Figures

1. **Event Timeline** - Daily events with rolling mean and change points
2. **Cause Distribution** - Section 150 vs network pie/bar comparison
3. **Inter-arrival Analysis** - Time between failures distribution
4. **Cyclical Patterns** - Hourly, daily, monthly, yearly trends
5. **Similar Sections** - Comparison with 10 similar sections
6. **PMU Characteristics** - Radar chart of key attributes
7. **Cumulative Events** - Growth curve vs expected rate

## Reports

- **Executive Summary**: 2-page overview for management
- **Technical Report**: Detailed statistical methodology and results

## Dependencies

Uses the same virtual environment as the main EDA analysis:
- pandas, numpy, scipy, statsmodels
- scikit-learn (similarity matching)
- plotly, matplotlib, seaborn (visualizations)
- kaleido (static image export)

## Relationship to EDA

This analysis builds on the main EDA work in `../EDA/`:
- Reuses `data_loader.py` for data loading
- Reuses `temporal.py` for time series analysis
- Follows the same code conventions and structure
