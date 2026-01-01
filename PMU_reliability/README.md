# PMU Reliability Framework

A production-ready Python package for power grid PMU (Phasor Measurement Unit) disturbance analysis and multi-dimensional risk scoring.

## Features

- **Data Loading**: Load and validate PMU disturbance data from Excel
- **Risk Scoring**: Multi-dimensional risk framework combining frequency, trend, MTBF, age, and recency
- **Temporal Analysis**: Anomaly detection, pattern mining, and clustering analysis
- **Spatial Analysis**: Geographic clustering and similarity-based section comparison
- **Visualization**: Publication-quality figures for research papers
- **Reproducibility**: Comprehensive test suite validating against known results

## Installation

```bash
# Navigate to project directory
cd PMU_reliability

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Analyze a Specific Section

```python
from src.data_loader import load_pmu_disturbance_data, get_section_events, calculate_event_statistics
from src.risk_scorer import PMURiskScorer
from src.temporal_analysis import TemporalAnalyzer

# Load data
pmu_df, dist_df = load_pmu_disturbance_data('../data/PMU_disturbance.xlsx')

# Get Section 150 events
events = get_section_events(dist_df, 150)
stats = calculate_event_statistics(events)
print(f"Section 150: {stats['count']} events, MTBF: {stats['mtbf_days']:.1f} days")

# Calculate risk scores
scorer = PMURiskScorer(pmu_df, dist_df)
risk_results = scorer.calculate_risk_scores()
print(f"Section 150 Rank: #{scorer.get_section_risk(150)['rank']}")

# Analyze temporal patterns
analyzer = TemporalAnalyzer(events)
peaks = analyzer.calculate_peak_periods()
print(f"Peak Hour: {peaks['peak_hour']}:00")
```

### Run Full Network Analysis

```bash
python scripts/run_full_analysis.py
```

### Generate Section Report

```bash
python scripts/generate_section_report.py --section_id 150
```

### Create Publication Figures

```bash
python scripts/create_paper_figures.py
```

## Project Structure

```
PMU_reliability/
├── src/                    # Core modules
│   ├── data_loader.py      # Data loading and validation
│   ├── risk_scorer.py      # Multi-dimensional risk scoring
│   ├── temporal_analysis.py # Temporal pattern analysis
│   ├── spatial_analysis.py # Geographic and similarity analysis
│   └── visualization.py    # Publication-quality figures
├── tests/                  # Pytest test suite
│   ├── test_reproducibility.py  # Validation against known results
│   ├── test_data_loader.py      # Data loading unit tests
│   ├── test_risk_scorer.py      # Risk scorer unit tests
│   └── test_temporal_analysis.py # Temporal analysis unit tests
├── scripts/                # Executable analysis scripts
│   ├── run_full_analysis.py     # Full network analysis
│   ├── generate_section_report.py # Section-specific reports
│   └── create_paper_figures.py  # Publication figures
├── notebooks/              # Jupyter notebooks for exploration
├── outputs/                # Generated results
│   ├── figures/            # Visualization outputs
│   ├── reports/            # Markdown reports
│   └── results/            # CSV data outputs
└── docs/                   # Documentation
```

## Risk Scoring Methodology

The multi-dimensional risk score combines five components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Frequency | 35% | Event count (higher = more risk) |
| Trend | 25% | Are events increasing over time? |
| MTBF | 20% | Mean time between failures (lower = more risk) |
| Age | 10% | Equipment age in years |
| Recency | 10% | Days since last event (lower = more risk) |

All components are normalized to 0-100 scale, then combined using weighted sum.

## Validation

The framework has been validated against Section 150 analysis:

| Metric | Expected | Validated |
|--------|----------|-----------|
| Event Count | 301 | ✅ |
| MTBF | ~16.4 days | ✅ |
| Network Ratio | ~25x | ✅ |
| Risk Rank | #1 | ✅ |
| Peak Hour | 19:00 | ✅ |

Run tests with:
```bash
pytest tests/ -v
```

## Requirements

- Python 3.10+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```
@software{pmu_reliability_framework,
  title = {PMU Reliability Framework},
  author = {Power Grid Research Team},
  year = {2026},
  version = {1.0.0}
}
```
