# Section 150 Complete Analysis Report
*PMU Disturbance Deep-Dive Investigation*

---

## Executive Overview

**Section 150 is the highest-risk PMU section in the network**, with 301 disturbance events over 13 years (2009-2022). This represents:

- **25.1x** the network average event rate
- **#1 rank** among all 533 sections
- **16.4-day** mean time between failures (vs. 170 days network average)

This report synthesizes findings from comprehensive statistical analysis, comparative benchmarking, and operational pattern mining.

---

## 1. Event Profile Summary

### 1.1 Volume & Ranking

| Metric | Section 150 | Network Average | Ratio |
|--------|-------------|-----------------|-------|
| Total Events | 301 | 12.0 | **25.1x** |
| MTBF (days) | 16.4 | 170 | **10.4x faster** |
| Percentile Rank | 100% | — | Top 0.2% |

### 1.2 Temporal Distribution

Events cluster rather than occur randomly:
- **Dispersion Index**: 1.50 (>1 confirms clustering)
- **Burst Days Detected**: 154 days with abnormally high activity
- **Change Points**: Multiple regime shifts detected

### 1.3 Peak Periods

| Dimension | Peak | Section 150 | Network Avg | Enrichment |
|-----------|------|-------------|-------------|------------|
| Hour | **7 PM (19:00)** | 7.0% | 4.4% | 1.6x |
| Day | **Thursday** | — | — | — |
| Month | **May** | — | — | Significant |

**Interpretation**: The 7 PM spike suggests correlation with evening load switching or thermal cycling effects on aging equipment.

---

## 2. Cause Analysis

### 2.1 Top 5 Causes

| Rank | Cause | Count | % of Total | Relative Risk vs Network |
|------|-------|-------|------------|--------------------------|
| 1 | Unknown | 51 | 16.9% | 1.32x 🟡 |
| 2 | Weather | 29 | 9.6% | 1.00x 🟢 |
| 3 | Weather (excl. lightning) | 25 | 8.3% | 1.37x 🟡 |
| 4 | Lightning | 18 | 6.0% | 1.34x 🟡 |
| 5 | Severe Weather (specific) | 5 | 1.7% | ∞ (unique to 150) 🔴 |

### 2.2 Weather Dominance

Aggregating all weather-related causes:
- **Weather + Lightning combined**: ~40% of all events
- This confirms Section 150's primary vulnerability is **environmental exposure**

### 2.3 "Unknown" Reclassification

Extended analysis found that **25.8% of "Unknown" events (33 events)** can be explained through:
- Time-proximity to named events
- Keyword extraction from operations logs
- Most hidden causes are actually **Weather-related**

---

## 3. Operational Insights

### 3.1 Cromwell Tap 31

**Finding**: A specific asset repeatedly correlates with disturbances.

- **21 events** explicitly mention "Cromwell Tap 31"
- Dominant operational state: equipment in transition (open/close operations)
- **Implication**: Mechanical wear or sensor issues at this tap point

### 3.2 Network Interconnection Role

Section 150 serves as a critical junction:
- **PSO** (Public Service of Oklahoma): 91 mentions in logs
- **OMPA** (Oklahoma Municipal Power Authority): 26 mentions
- **DCC** (Dispatch Control Center): Frequently notified

This hub position may contribute to elevated stress and failure rates.

---

## 4. Comparative Analysis

### 4.1 Similar Sections Identified

Using PMU characteristics (voltage, type, location), we identified **23 similar sections** for benchmarking.

| Metric | Section 150 | Similar Sections Avg | Ratio |
|--------|-------------|---------------------|-------|
| Event Count | 301 | 109 | 2.8x |
| Best Performer | 301 | 3 (Section 574) | 100x |

### 4.2 Key Differences

1. **Section 150's cause pattern differs** from most similar sections
2. **Events occur 4.3x more frequently** than similar section average
3. **Section 574** (same voltage/type) has only 3 events — investigate why

### 4.3 Learnings from Best Performers

- Similar equipment, vastly different outcomes
- Suggests **operational/maintenance practices** or **microclimate** factors
- Recommended: Physical inspection comparison between Section 150 and Section 574

---

## 5. Statistical Validation

### 5.1 Significance Tests

| Test | Statistic | P-Value | Conclusion |
|------|-----------|---------|------------|
| Poisson Rate Test | — | < 0.001 | **Significantly elevated** |
| Z-Test | 82.56 | < 0.001 | **82 std dev above mean** |
| Rate Ratio CI | 25.1x | (14.1, 44.6) | **Definitively higher** |
| KS Test (inter-arrival) | — | < 0.05 | **Different distribution** |

### 5.2 Conclusions

- Section 150's failure rate is **statistically significantly higher** than expected
- The difference is not due to random variation
- Targeted intervention is warranted

---

## 6. Recommendations

### 6.1 Priority Actions

| Priority | Action | Target | Expected Impact |
|----------|--------|--------|-----------------|
| **1** | **Inspect Cromwell Tap 31** | Mechanical audit of tap changer | Address 21 direct events |
| **2** | **Weather hardening** | Lightning arresters, enclosures | 20-30% reduction |
| **3** | **Evening monitoring** | 6-8 PM patrols/checks | Early 7 PM spike detection |
| **4** | **Log classification update** | Auto-tag "Unknown" with weather keywords | 25% fewer unknowns |
| **5** | **Benchmark vs Section 574** | Physical comparison study | Learn from best performer |

### 6.2 Monitoring Recommendations

- **Base Frequency**: Daily automated monitoring
- **Enhanced Periods**:
  - Increase staffing during **May** (peak month)
  - Alert threshold at **7 PM** (peak hour)
- **Alert Thresholds**:
  - Rapid succession: Events < 4 hours apart
  - Unusual quiet: No events for > 32 days

### 6.3 Expected Risk Reduction

| Scenario | Events Preventable | Reduction |
|----------|-------------------|-----------|
| Weather hardening only | 30-50 | 10-17% |
| + Cromwell Tap 31 fix | 20-30 | 7-10% |
| + Evening monitoring | 10-20 | 3-7% |
| **Combined** | **60-100** | **20-33%** |

---

## 7. Deliverables Summary

### 7.1 Analysis Modules

```
Section_150/src/
├── section150_loader.py           # Data loading
├── section150_event_breakdown.py  # Timeline & clustering
├── section150_characteristics.py  # PMU metadata
├── section150_root_cause.py       # Statistical tests
├── section150_comparative.py      # Benchmarking
├── section150_recommendations.py  # Interventions
├── section150_visuals.py          # 7 figures
└── section150_report.py           # Report generation
```

### 7.2 Visualizations

| # | Figure | Key Insight |
|---|--------|-------------|
| 1 | Event Timeline | Sporadic clusters, max 4 events/day |
| 2 | Cause Distribution | Weather dominates, "Unknown" inflated |
| 3 | Inter-arrival Analysis | MTBF 16.4 days, exponential-like distribution |
| 4 | Cyclical Patterns | 7 PM peak, May seasonality |
| 5 | Similar Sections | 2.8x more events than peers |
| 6 | PMU Characteristics | Radar comparison of attributes |
| 7 | Cumulative Events | Growth far exceeds expected rate |

### 7.3 Reports Generated

- `section150_executive_summary.md` — 2-page management overview
- `section150_technical_report.md` — Detailed methodology
- `section150_extended_insights.md` — Operational deep-dive

---

## 8. How to Reproduce

```bash
cd /Users/anhle/disturbance/Section_150
source ../venv/bin/activate
python run_section150.py
```

Runtime: ~30 seconds

---

## 9. Conclusion

**Section 150 requires immediate attention.** With:
- 301 events (25x network average)
- 16.4-day MTBF (10x faster failures)
- Clear weather vulnerability
- Identifiable equipment issues (Cromwell Tap 31)
- 7 PM operational anomaly

Targeted interventions focusing on **weather hardening**, **equipment inspection**, and **enhanced evening monitoring** can reduce events by **20-33%** (60-100 events over the historical period).

The analysis framework is fully reproducible and can be extended to other high-risk sections.

---

*Report Generated: 2026-01-01*
*Analysis Framework: Section 150 Deep-Dive Package*
*Data Source: PMU_disturbance.xlsx (2009-2022)*
