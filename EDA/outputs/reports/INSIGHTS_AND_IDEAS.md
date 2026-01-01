# PMU Disturbance Analysis: Insights & Further Analysis Ideas

## 📊 Generated Visualizations

All figures saved to: `outputs/figures/static/`

| Figure | Description |
|--------|-------------|
| [01_daily_disturbances_timeline.png](file:///Users/anhle/disturbance/outputs/figures/static/01_daily_disturbances_timeline.png) | Daily events with 30-day moving average |
| [02_monthly_disturbances.png](file:///Users/anhle/disturbance/outputs/figures/static/02_monthly_disturbances.png) | Monthly aggregation with trend |
| [03_cyclical_patterns.png](file:///Users/anhle/disturbance/outputs/figures/static/03_cyclical_patterns.png) | Hour/Day/Month/Year patterns |
| [04_top_causes_bar.png](file:///Users/anhle/disturbance/outputs/figures/static/04_top_causes_bar.png) | Top 15 disturbance causes |
| [05_pareto_causes.png](file:///Users/anhle/disturbance/outputs/figures/static/05_pareto_causes.png) | Pareto (80/20) analysis |
| [06_cause_pie_chart.png](file:///Users/anhle/disturbance/outputs/figures/static/06_cause_pie_chart.png) | Cause distribution pie chart |
| [07_risk_distribution.png](file:///Users/anhle/disturbance/outputs/figures/static/07_risk_distribution.png) | Risk score histogram & categories |
| [08_top_risk_sections.png](file:///Users/anhle/disturbance/outputs/figures/static/08_top_risk_sections.png) | Top 20 highest risk sections |
| [09_risk_vs_frequency.png](file:///Users/anhle/disturbance/outputs/figures/static/09_risk_vs_frequency.png) | Risk vs historical frequency |
| [10_geographic_disturbances.png](file:///Users/anhle/disturbance/outputs/figures/static/10_geographic_disturbances.png) | PMU locations by disturbance count |
| [11_geographic_clusters.png](file:///Users/anhle/disturbance/outputs/figures/static/11_geographic_clusters.png) | K-means geographic clusters |
| [12_reliability_metrics.png](file:///Users/anhle/disturbance/outputs/figures/static/12_reliability_metrics.png) | MTBF distribution |
| [13_anomaly_detection.png](file:///Users/anhle/disturbance/outputs/figures/static/13_anomaly_detection.png) | Anomaly days highlighted |
| [14_heatmap_hour_day.png](file:///Users/anhle/disturbance/outputs/figures/static/14_heatmap_hour_day.png) | Hour vs Day heatmap |
| [15_heatmap_month_year.png](file:///Users/anhle/disturbance/outputs/figures/static/15_heatmap_month_year.png) | Month vs Year heatmap |
| [16_voltage_distribution.png](file:///Users/anhle/disturbance/outputs/figures/static/16_voltage_distribution.png) | Disturbances by voltage level |
| [17_forecast_90days.png](file:///Users/anhle/disturbance/outputs/figures/static/17_forecast_90days.png) | 90-day forecast with CI |

---

## 🔍 Key Insights from Analysis

### 1. **Temporal Patterns**
- **Peak Activity**: 10:00 AM is the peak hour (work hours operations)
- **Weekly Pattern**: Mondays have highest disturbances (1,497 events) - likely detection/reporting after weekend
- **Seasonal Pattern**: May has the most disturbances (1,526 events) - spring storm season
- **No Long-term Trend**: Mann-Kendall test shows no significant increasing/decreasing trend (p=0.14)

### 2. **Cause Analysis**
- **Unknown Causes (12.9%)**: Biggest category → **Opportunity to improve root cause documentation**
- **Weather-related (20%+)**: Weather + Lightning + "Weather excluding lightning" combined
- **Long-tail Distribution**: 3,066 unique cause descriptions for 80% coverage → **Standardize cause codes**

### 3. **Risk Concentration**
- **Section 150**: Highest risk (64.8) with 301 historical events - **Immediate attention needed**
- **Only 1 High-risk section**: Most sections (328) are medium risk
- **Risk vs Frequency**: Strong correlation but not perfect → age and recency matter too

### 4. **Spatial Patterns**
- **5 DBSCAN clusters** identified geographically
- **Hotspots exist**: Some locations have significantly more events
- **Network Analysis**: Section 1385 has highest betweenness centrality (critical junction)

### 5. **Reliability Metrics**
- **Average MTBF**: ~9,872 hours (~411 days) between failures
- **High variability**: Some sections fail frequently, others rarely

---

## 💡 Ideas for Further Analysis

### **Immediate Priority (High Value)**

1. **Deep-Dive on Section 150**
   - Why does it have 301 events (highest)?
   - What are the specific causes?
   - Equipment age, voltage level, geographic factors?
   - Targeted maintenance strategy

2. **Weather Correlation Study**
   - Correlate with NOAA weather data (storms, temperature, wind)
   - Build weather-based early warning system
   - Identify weather thresholds that trigger events

3. **Unknown Cause Investigation**
   - Analyze patterns in "Unknown" causes
   - Cross-reference with Operations field
   - Improve classification/reporting processes

### **Medium Priority (Strategic Value)**

4. **Cascading Failure Analysis**
   - Identify sequential patterns (A→B within hours)
   - Build cascade prediction model
   - Network propagation analysis

5. **PMU Age vs Failure Rate (Bathtub Curve)**
   - Analyze infant mortality vs wear-out phases
   - Determine optimal replacement timing
   - Calculate remaining useful life

6. **Voltage Level Deep-Dive**
   - Compare failure rates across voltage levels
   - Statistical test for significant differences
   - Voltage-specific maintenance strategies

7. **Geographic Hotspot Prevention**
   - Why do certain areas cluster?
   - Common infrastructure, weather exposure?
   - Targeted hardening investments

### **Advanced Analysis (Research Value)**

8. **Machine Learning Prediction**
   - Build ML model to predict next-day disturbance probability
   - Features: weather forecast, PMU age, recent events, day-of-week
   - Random Forest, XGBoost, or LSTM for time series

9. **Survival Analysis Enhancement**
   - Kaplan-Meier curves by voltage level, type, region
   - Cox regression with more covariates
   - Time-varying hazard models

10. **Operations Field Decoding**
    - Parse and analyze the Operations column
    - Extract repair times, crew information
    - Calculate true MTTR (Mean Time To Repair)

11. **Cost-Benefit Analysis**
    - Estimate cost per disturbance
    - Prioritize investments by ROI
    - Maintenance optimization model

12. **Real-time Dashboard**
    - Build interactive Plotly/Dash dashboard
    - Live monitoring of risk scores
    - Alert system for anomalous patterns

---

## 🎯 Recommended Next Steps

| Priority | Action | Expected Outcome |
|----------|--------|------------------|
| 🔴 High | Investigate Section 150 | Reduce highest-risk section events |
| 🔴 High | Weather data integration | Early warning capability |
| 🟡 Medium | Standardize cause codes | Better root cause analysis |
| 🟡 Medium | Age-based maintenance | Optimized replacement schedule |
| 🟢 Low | ML prediction model | Proactive resource allocation |
| 🟢 Low | Real-time dashboard | Operational visibility |

---

## 📈 Suggested Analyses to Run Next

```python
# 1. Deep-dive on Section 150
section_150 = disturbance_df[disturbance_df['SectionID'] == 150]
print(section_150['Cause'].value_counts().head(10))

# 2. PMU Age Analysis
pmu_df['Age_Years'] = (pd.Timestamp.now() - pmu_df['InService']).dt.days / 365.25
merged = pmu_df.merge(disturbance_df.groupby('SectionID').size().reset_index(name='Events'))
plt.scatter(merged['Age_Years'], merged['Events'])

# 3. Weather correlation (if weather data available)
# Join with NOAA data on date and location

# 4. Sequential pattern deep-dive
from src.causality import detect_sequential_patterns
seq = detect_sequential_patterns(disturbance_df, 'Timestamp', 'Cause', window_days=1)
print(seq.head(20))
```

---

*Generated: 2025-12-31*
*Analysis based on 9,369 disturbance events across 533 PMU sections (2009-2022)*
