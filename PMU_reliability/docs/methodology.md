# Risk Scoring Methodology

## Overview

The PMU Reliability Framework implements a multi-dimensional risk scoring system that evaluates PMU sections based on five key components. This document provides the mathematical foundation and rationale for each component.

## Risk Score Formula

The composite risk score is calculated as:

```
Risk Score = Σ (weight_i × normalized_score_i)
```

Where:
- `weight_i` is the component weight (sums to 1.0)
- `normalized_score_i` is the normalized component score (0-100)

## Component Definitions

### 1. Frequency Score (Default Weight: 35%)

**Definition**: Total count of disturbance events for the section.

**Formula**:
```
frequency_raw = count(events)
frequency_score = normalize(frequency_raw) × 100
```

**Rationale**: Sections with more events are inherently higher risk because they have demonstrated a pattern of failures.

### 2. Trend Score (Default Weight: 25%)

**Definition**: The slope of event frequency over time, indicating whether events are increasing or decreasing.

**Formula**:
```
trend_raw = slope(linear_regression(daily_counts ~ time)) × 1000
trend_score = normalize(trend_raw) × 100
```

**Rationale**: A positive trend suggests degrading equipment or worsening conditions, warranting higher risk classification.

### 3. MTBF Score (Default Weight: 20%)

**Definition**: Mean Time Between Failures - the average number of days between consecutive events.

**Formula**:
```
mtbf_raw = mean(inter_arrival_times)
mtbf_score = (1 - normalize(mtbf_raw)) × 100  # Inverted
```

**Rationale**: Lower MTBF indicates more frequent failures. The score is inverted so that lower MTBF results in higher risk.

### 4. Age Score (Default Weight: 10%)

**Definition**: Age of the PMU equipment in years since installation.

**Formula**:
```
age_raw = (current_date - InService_date).days / 365.25
age_score = normalize(age_raw) × 100
```

**Rationale**: Older equipment may be more prone to failures due to wear and tear.

### 5. Recency Score (Default Weight: 10%)

**Definition**: Days since the most recent disturbance event.

**Formula**:
```
recency_raw = (current_date - last_event_date).days
recency_score = (1 - normalize(recency_raw)) × 100  # Inverted
```

**Rationale**: Recent events suggest ongoing issues. The score is inverted so that more recent events result in higher risk.

## Normalization

All raw scores are normalized using min-max normalization:

```
normalized = (value - min) / (max - min) × 100
```

This ensures all components are on the same 0-100 scale before weighting.

## Risk Categories

Based on the composite score, sections are categorized as:

| Category | Score Range | Description |
|----------|-------------|-------------|
| High | ≥ 66.67 | Immediate attention required |
| Medium | 33.33 - 66.66 | Monitor closely |
| Low | < 33.33 | Acceptable risk level |

## Default Weights

The default weights were determined based on:
- Domain expertise in power grid reliability
- Correlation analysis with historical failures
- Sensitivity analysis on predictive accuracy

```python
DEFAULT_WEIGHTS = {
    'frequency': 0.35,  # Most predictive of future events
    'trend': 0.25,      # Important for proactive maintenance
    'mtbf': 0.20,       # Reliability metric
    'age': 0.10,        # Equipment lifecycle consideration
    'recency': 0.10     # Recent activity indicator
}
```

## Customizing Weights

Users can provide custom weights based on their specific operational priorities:

```python
custom_weights = {
    'frequency': 0.50,  # Prioritize event count
    'trend': 0.20,
    'mtbf': 0.15,
    'age': 0.10,
    'recency': 0.05
}

scorer = PMURiskScorer(pmu_df, dist_df, weights=custom_weights)
```

Note: Custom weights must sum to 1.0.

## Validation

The methodology was validated against:
1. **Section 150**: Known highest-risk section correctly identified as #1
2. **Historical outcomes**: Correlation with actual maintenance actions
3. **Expert review**: Alignment with domain expert rankings

## Limitations

- Trend analysis requires minimum 3 events for meaningful calculation
- MTBF requires minimum 2 events (need interval between events)
- Age calculation requires InService date in PMU data
- Geographic factors not directly incorporated (handled by similarity analysis)

## References

1. IEEE C37.118 - Standard for Synchrophasor Measurements
2. NERC Reliability Standards for PMU Performance
3. Power Grid Reliability Assessment Guidelines
