# API Reference

This document provides detailed documentation for all modules in the PMU Reliability Framework.

---

## Module: `data_loader`

**File**: `src/data_loader.py`

### Functions

#### `load_pmu_disturbance_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]`
Load PMU and Disturbance data from Excel file.

**Parameters:**
- `filepath`: Path to PMU_disturbance.xlsx file

**Returns:**
- `pmu_df`: DataFrame with PMU installation data (533 records)
- `dist_df`: DataFrame with disturbance events (9,369 records)

**Example:**
```python
from data_loader import load_pmu_disturbance_data
pmu_df, dist_df = load_pmu_disturbance_data('data/PMU_disturbance.xlsx')
```

---

#### `get_section_events(dist_df: pd.DataFrame, section_id: int) -> pd.DataFrame`
Extract all disturbance events for a specific section.

**Parameters:**
- `dist_df`: Disturbance DataFrame
- `section_id`: Section ID to filter (e.g., 150)

**Returns:**
- DataFrame containing only events for the specified section

**Example:**
```python
section_150_events = get_section_events(dist_df, 150)
print(f"Section 150 has {len(section_150_events)} events")  # 301 events
```

---

#### `calculate_event_statistics(events_df: pd.DataFrame) -> dict`
Calculate basic statistics for a set of events.

**Parameters:**
- `events_df`: DataFrame of events (for one section)

**Returns:**
- Dictionary with keys:
  - `count`: Number of events
  - `mtbf_days`: Mean time between failures in days
  - `first_event`: Date of first event
  - `last_event`: Date of last event

**Example:**
```python
stats = calculate_event_statistics(section_150_events)
print(f"MTBF: {stats['mtbf_days']:.1f} days")  # ~16.4 days
```

---

## Module: `risk_scorer`

**File**: `src/risk_scorer.py`

### Class: `PMURiskScorer`

Multi-dimensional risk scoring for PMU sections.

#### `__init__(self, pmu_df, dist_df, weights=None)`
Initialize the risk scorer.

**Parameters:**
- `pmu_df`: PMU installation DataFrame
- `dist_df`: Disturbance events DataFrame
- `weights`: Optional dict with component weights (default: frequency=35%, trend=25%, mtbf=20%, age=10%, recency=10%)

**Example:**
```python
from risk_scorer import PMURiskScorer

scorer = PMURiskScorer(pmu_df, dist_df)
# Or with custom weights:
scorer = PMURiskScorer(pmu_df, dist_df, weights={
    'frequency': 0.50,
    'trend': 0.20,
    'mtbf': 0.15,
    'age': 0.10,
    'recency': 0.05
})
```

---

#### `calculate_risk_scores(self) -> pd.DataFrame`
Calculate risk scores for all sections.

**Returns:**
- DataFrame with columns:
  - `SectionID`: Section identifier
  - `risk_score`: Composite risk score (0-100)
  - `rank`: Risk ranking (1 = highest risk)
  - `category`: Risk category (Low/Medium/High)
  - `frequency_score`, `trend_score`, `mtbf_score`, `age_score`, `recency_score`: Component scores

**Example:**
```python
results = scorer.calculate_risk_scores()
print(f"Section 150 rank: #{results[results['SectionID']==150]['rank'].values[0]}")  # #1
```

---

## Module: `temporal_analysis`

**File**: `src/temporal_analysis.py`

### Class: `TemporalAnalyzer`

#### `__init__(self, events_df: pd.DataFrame)`
Initialize with event data.

---

#### `detect_anomalies(self, method='iqr') -> pd.DataFrame`
Detect days with anomalously high event counts.

**Parameters:**
- `method`: 'iqr' or 'zscore'

**Returns:**
- DataFrame of anomalous days with event counts

---

#### `calculate_hourly_pattern(self) -> pd.Series`
Calculate event distribution by hour of day.

**Returns:**
- Series with hours (0-23) as index and event counts as values

---

#### `calculate_peak_periods(self) -> dict`
Identify peak hour, day, and month.

**Returns:**
- Dictionary with keys: `peak_hour`, `peak_day`, `peak_month`

**Example:**
```python
from temporal_analysis import TemporalAnalyzer

analyzer = TemporalAnalyzer(section_150_events)
peaks = analyzer.calculate_peak_periods()
print(f"Peak hour: {peaks['peak_hour']}:00")  # 19:00 for Section 150
```

---

#### `test_clustering(self) -> dict`
Test for temporal clustering using dispersion index.

**Returns:**
- Dictionary with keys: `dispersion_index`, `is_clustered`
- `is_clustered` is True if dispersion_index > 1.0

---

## Module: `spatial_analysis`

**File**: `src/spatial_analysis.py`

### Class: `SpatialAnalyzer`

#### `cluster_geographic(self, eps=0.5) -> pd.DataFrame`
Perform DBSCAN clustering on geographic coordinates.

**Parameters:**
- `eps`: DBSCAN epsilon parameter

**Returns:**
- PMU DataFrame with added `cluster_id` column

---

#### `find_similar_sections(self, section_id: int, k=10) -> pd.DataFrame`
Find k sections most similar to the target section.

**Parameters:**
- `section_id`: Target section ID
- `k`: Number of similar sections to return

**Returns:**
- DataFrame of k most similar sections with similarity scores

---

## Module: `visualization`

**File**: `src/visualization.py`

### Functions

#### `plot_event_timeline(events, section_id, save_path)`
Create event timeline with trend line.

#### `plot_cause_distribution(events, section_id, save_path)`
Create cause distribution charts (bar + pie).

#### `plot_temporal_heatmap(events, section_id, save_path)`
Create hour/day/month heatmaps.

#### `plot_risk_distribution(risk_results, save_path)`
Create risk score histogram and category distribution.

#### `create_section_report_figures(section_id, output_dir)`
Generate all standard figures for a section analysis.

**Parameters:**
- `section_id`: Section to analyze
- `output_dir`: Directory to save figures

**Returns:**
- List of paths to generated figures

---

## Quick Reference

### Load Data
```python
from data_loader import load_pmu_disturbance_data, get_section_events, calculate_event_statistics

pmu_df, dist_df = load_pmu_disturbance_data('data/PMU_disturbance.xlsx')
events = get_section_events(dist_df, 150)
stats = calculate_event_statistics(events)
```

### Calculate Risk Scores
```python
from risk_scorer import PMURiskScorer

scorer = PMURiskScorer(pmu_df, dist_df)
results = scorer.calculate_risk_scores()
top_10 = results.head(10)
```

### Analyze Temporal Patterns
```python
from temporal_analysis import TemporalAnalyzer

analyzer = TemporalAnalyzer(events)
peaks = analyzer.calculate_peak_periods()
clustering = analyzer.test_clustering()
```

### Generate Visualizations
```python
from visualization import create_section_report_figures

figures = create_section_report_figures(150, 'outputs/figures/')
```

---

*API Reference v1.0 - PMU Reliability Framework*
