#!/usr/bin/env python3
"""
PMU Disturbance Analysis - Visualization Generator
Generates all analysis figures.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import config
from src import data_loader, temporal, causality, spatial, predictive

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

FIGURE_DIR = os.path.join(config.OUTPUT_DIR, 'figures', 'static')
os.makedirs(FIGURE_DIR, exist_ok=True)

def save_figure(fig, name):
    """Save figure to output directory."""
    path = os.path.join(FIGURE_DIR, f'{name}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   ✅ Saved: {name}.png")

# Load data
print("Loading data...")
pmu_df, disturbance_df, merged_df = data_loader.load_all_data(
    config.EXCEL_FILE, config.PMU_SHEET, config.DISTURBANCE_SHEET
)

datetime_col = 'Timestamp'
cause_col = 'Cause'

# Load risk scores
risk_scores = pd.read_csv(config.RISK_SCORES)

print(f"\n{'='*60}")
print("📊 GENERATING VISUALIZATIONS")
print(f"{'='*60}")

# ============================================================
# 1. TEMPORAL ANALYSIS FIGURES
# ============================================================
print("\n1️⃣  Temporal Analysis Figures...")

# Daily disturbance counts over time
ts = temporal.aggregate_disturbances_by_time(disturbance_df, datetime_col, freq='D')

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts.index, ts.values, alpha=0.5, linewidth=0.5, label='Daily Count')
ax.plot(ts.index, ts.rolling(30).mean(), color='red', linewidth=2, label='30-day Moving Average')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Disturbances', fontsize=12)
ax.set_title('Daily Disturbance Events Over Time (2009-2022)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
save_figure(fig, '01_daily_disturbances_timeline')

# Monthly aggregation
ts_monthly = temporal.aggregate_disturbances_by_time(disturbance_df, datetime_col, freq='M')

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(ts_monthly.index, ts_monthly.values, width=25, alpha=0.7, color='steelblue')
ax.plot(ts_monthly.index, ts_monthly.rolling(12).mean(), color='red', linewidth=2, label='12-month MA')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Monthly Disturbances', fontsize=12)
ax.set_title('Monthly Disturbance Counts with Trend', fontsize=14, fontweight='bold')
ax.legend()
save_figure(fig, '02_monthly_disturbances')

# Cyclical patterns
patterns = temporal.extract_cyclical_patterns(disturbance_df, datetime_col)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Hourly pattern
ax = axes[0, 0]
hours = range(24)
hourly_counts = [patterns['hourly'].get(h, 0) for h in hours]
ax.bar(hours, hourly_counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Disturbances')
ax.set_title('Disturbances by Hour of Day', fontweight='bold')
ax.set_xticks(range(0, 24, 2))

# Daily pattern
ax = axes[0, 1]
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_counts = [patterns['daily'].get(d, 0) for d in days]
colors = ['coral' if d in ['Saturday', 'Sunday'] else 'steelblue' for d in days]
ax.bar(range(7), daily_counts, color=colors, alpha=0.7)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Number of Disturbances')
ax.set_title('Disturbances by Day of Week', fontweight='bold')
ax.set_xticks(range(7))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# Monthly pattern
ax = axes[1, 0]
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
monthly_counts = [patterns['monthly'].get(m, 0) for m in months]
ax.bar(range(12), monthly_counts, color='steelblue', alpha=0.7)
ax.set_xlabel('Month')
ax.set_ylabel('Number of Disturbances')
ax.set_title('Disturbances by Month', fontweight='bold')
ax.set_xticks(range(12))
ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])

# Yearly trend
ax = axes[1, 1]
yearly = disturbance_df.copy()
yearly['Year'] = pd.to_datetime(yearly[datetime_col]).dt.year
yearly_counts = yearly.groupby('Year').size()
ax.bar(yearly_counts.index, yearly_counts.values, color='steelblue', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Disturbances')
ax.set_title('Disturbances by Year', fontweight='bold')
ax.axhline(yearly_counts.mean(), color='red', linestyle='--', label=f'Mean: {yearly_counts.mean():.0f}')
ax.legend()

plt.tight_layout()
save_figure(fig, '03_cyclical_patterns')

# ============================================================
# 2. CAUSE ANALYSIS FIGURES
# ============================================================
print("\n2️⃣  Cause Analysis Figures...")

cause_dist = causality.analyze_cause_distribution(disturbance_df, cause_col)

# Top 15 causes bar chart
fig, ax = plt.subplots(figsize=(12, 8))
top_causes = cause_dist.head(15)
y_pos = range(len(top_causes))
ax.barh(y_pos, top_causes['Count'], color='steelblue', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels([c[:40] + '...' if len(c) > 40 else c for c in top_causes.index])
ax.set_xlabel('Number of Disturbances', fontsize=12)
ax.set_title('Top 15 Disturbance Causes', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add percentage labels
for i, (count, pct) in enumerate(zip(top_causes['Count'], top_causes['Percentage'])):
    ax.text(count + 10, i, f'{pct:.1f}%', va='center', fontsize=9)

plt.tight_layout()
save_figure(fig, '04_top_causes_bar')

# Pareto chart
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

top20 = cause_dist.head(20)
x_pos = range(len(top20))

ax1.bar(x_pos, top20['Count'], color='steelblue', alpha=0.7, label='Count')
ax2.plot(x_pos, top20['Cumulative_Percentage'], color='red', marker='o', linewidth=2, label='Cumulative %')
ax2.axhline(80, color='green', linestyle='--', alpha=0.7, label='80% threshold')

ax1.set_xlabel('Cause Category', fontsize=12)
ax1.set_ylabel('Count', fontsize=12, color='steelblue')
ax2.set_ylabel('Cumulative Percentage', fontsize=12, color='red')
ax1.set_title('Pareto Analysis of Disturbance Causes', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([str(i+1) for i in x_pos])

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

save_figure(fig, '05_pareto_causes')

# Cause pie chart (top 10 + Other)
fig, ax = plt.subplots(figsize=(10, 10))
top10 = cause_dist.head(10).copy()
other_count = cause_dist.iloc[10:]['Count'].sum()
top10.loc['All Other Causes'] = {'Count': other_count, 'Percentage': 100 - top10['Percentage'].sum()}

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top10)))
wedges, texts, autotexts = ax.pie(top10['Count'], labels=None, autopct='%1.1f%%',
                                   colors=colors, pctdistance=0.75)
ax.legend(wedges, [c[:30] + '...' if len(c) > 30 else c for c in top10.index],
          loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
ax.set_title('Distribution of Disturbance Causes', fontsize=14, fontweight='bold')

save_figure(fig, '06_cause_pie_chart')

# ============================================================
# 3. RISK ANALYSIS FIGURES
# ============================================================
print("\n3️⃣  Risk Analysis Figures...")

# Risk score distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(risk_scores['Risk_Score_0_100'], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(risk_scores['Risk_Score_0_100'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {risk_scores["Risk_Score_0_100"].mean():.1f}')
ax.axvline(risk_scores['Risk_Score_0_100'].median(), color='green', linestyle='--',
           linewidth=2, label=f'Median: {risk_scores["Risk_Score_0_100"].median():.1f}')
ax.set_xlabel('Risk Score (0-100)', fontsize=12)
ax.set_ylabel('Number of Sections', fontsize=12)
ax.set_title('Distribution of Risk Scores', fontsize=14, fontweight='bold')
ax.legend()

# Risk category counts
ax = axes[1]
category_counts = risk_scores['Risk_Category'].value_counts()
category_order = ['Critical', 'High', 'Medium', 'Low']
category_counts = category_counts.reindex([c for c in category_order if c in category_counts.index])
colors = {'Critical': 'darkred', 'High': 'red', 'Medium': 'orange', 'Low': 'green'}
ax.bar(category_counts.index, category_counts.values, 
       color=[colors.get(c, 'gray') for c in category_counts.index], alpha=0.8)
ax.set_xlabel('Risk Category', fontsize=12)
ax.set_ylabel('Number of Sections', fontsize=12)
ax.set_title('Sections by Risk Category', fontsize=14, fontweight='bold')

for i, v in enumerate(category_counts.values):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
save_figure(fig, '07_risk_distribution')

# Top 20 highest risk sections
fig, ax = plt.subplots(figsize=(12, 8))
top20_risk = risk_scores.head(20)
y_pos = range(len(top20_risk))
colors = ['darkred' if c == 'Critical' else 'red' if c == 'High' else 'orange' 
          for c in top20_risk['Risk_Category']]
ax.barh(y_pos, top20_risk['Risk_Score_0_100'], color=colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Section {s}" for s in top20_risk['SectionID']])
ax.set_xlabel('Risk Score (0-100)', fontsize=12)
ax.set_title('Top 20 Highest Risk PMU Sections', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for i, (score, freq) in enumerate(zip(top20_risk['Risk_Score_0_100'], top20_risk['Historical_Frequency'])):
    ax.text(score + 0.5, i, f'{score:.1f} ({freq} events)', va='center', fontsize=9)

plt.tight_layout()
save_figure(fig, '08_top_risk_sections')

# Risk score vs Historical frequency scatter
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(risk_scores['Historical_Frequency'], risk_scores['Risk_Score_0_100'],
                     c=risk_scores['Risk_Score_0_100'], cmap='RdYlGn_r', alpha=0.6, s=50)
ax.set_xlabel('Historical Frequency (Number of Events)', fontsize=12)
ax.set_ylabel('Risk Score (0-100)', fontsize=12)
ax.set_title('Risk Score vs Historical Disturbance Frequency', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Risk Score')

# Annotate top 5
for _, row in risk_scores.head(5).iterrows():
    ax.annotate(f"Section {int(row['SectionID'])}", 
                (row['Historical_Frequency'], row['Risk_Score_0_100']),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

save_figure(fig, '09_risk_vs_frequency')

# ============================================================
# 4. SPATIAL ANALYSIS FIGURES
# ============================================================
print("\n4️⃣  Spatial Analysis Figures...")

# Geographic scatter plot
lat_col = 'Latitude'
lon_col = 'Longitude'

if lat_col in pmu_df.columns and lon_col in pmu_df.columns:
    # Merge with disturbance counts
    dist_counts = disturbance_df.groupby('SectionID').size().reset_index(name='Disturbance_Count')
    pmu_with_dist = pmu_df.merge(dist_counts, on='SectionID', how='left')
    pmu_with_dist['Disturbance_Count'] = pmu_with_dist['Disturbance_Count'].fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(pmu_with_dist[lon_col], pmu_with_dist[lat_col],
                         c=pmu_with_dist['Disturbance_Count'], cmap='RdYlGn_r',
                         s=pmu_with_dist['Disturbance_Count'] + 10, alpha=0.6)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('PMU Locations Colored by Disturbance Count', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Disturbance Count')
    save_figure(fig, '10_geographic_disturbances')
    
    # Clustering visualization
    clustered = spatial.perform_kmeans_clustering(pmu_df, lat_col, lon_col, n_clusters=8)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(clustered[lon_col], clustered[lat_col],
                         c=clustered['Cluster'], cmap='tab10', s=50, alpha=0.7)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('PMU Geographic Clusters (K-Means, k=8)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster ID')
    save_figure(fig, '11_geographic_clusters')

# ============================================================
# 5. RELIABILITY METRICS FIGURES
# ============================================================
print("\n5️⃣  Reliability Metrics Figures...")

mtbf_df = causality.calculate_mtbf_mttr(disturbance_df, datetime_col)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MTBF distribution
ax = axes[0]
mtbf_valid = mtbf_df['MTBF_hours'].dropna()
ax.hist(mtbf_valid, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(mtbf_valid.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mtbf_valid.mean():.0f} hrs')
ax.set_xlabel('MTBF (Hours)', fontsize=12)
ax.set_ylabel('Number of Sections', fontsize=12)
ax.set_title('Distribution of Mean Time Between Failures', fontsize=14, fontweight='bold')
ax.legend()

# Failure count distribution
ax = axes[1]
ax.hist(mtbf_df['Failure_Count'], bins=50, color='coral', alpha=0.7, edgecolor='white')
ax.axvline(mtbf_df['Failure_Count'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mtbf_df["Failure_Count"].mean():.1f}')
ax.set_xlabel('Number of Failures per Section', fontsize=12)
ax.set_ylabel('Number of Sections', fontsize=12)
ax.set_title('Distribution of Failure Counts by Section', fontsize=14, fontweight='bold')
ax.legend()

plt.tight_layout()
save_figure(fig, '12_reliability_metrics')

# ============================================================
# 6. ANOMALY DETECTION FIGURES
# ============================================================
print("\n6️⃣  Anomaly Detection Figures...")

anomalies_zscore = temporal.detect_anomalies_zscore(ts, threshold=3.0)
anomalies_iqr = temporal.detect_anomalies_iqr(ts, multiplier=1.5)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts.index, ts.values, alpha=0.5, linewidth=0.5, label='Daily Count', color='steelblue')
ax.scatter(ts.index[anomalies_zscore], ts[anomalies_zscore], color='red', s=30, 
           label=f'Z-score Anomalies ({anomalies_zscore.sum()})', zorder=5)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Disturbances', fontsize=12)
ax.set_title('Anomaly Detection: Days with Unusual Disturbance Activity', fontsize=14, fontweight='bold')
ax.legend()
save_figure(fig, '13_anomaly_detection')

# ============================================================
# 7. HEATMAP FIGURES
# ============================================================
print("\n7️⃣  Heatmap Figures...")

# Hour vs Day of Week heatmap
disturbance_df_copy = disturbance_df.copy()
disturbance_df_copy['Hour'] = pd.to_datetime(disturbance_df_copy[datetime_col]).dt.hour
disturbance_df_copy['DayOfWeek'] = pd.to_datetime(disturbance_df_copy[datetime_col]).dt.dayofweek

heatmap_data = disturbance_df_copy.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Disturbances'})
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Day of Week', fontsize=12)
ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
ax.set_title('Disturbance Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold')
save_figure(fig, '14_heatmap_hour_day')

# Month vs Year heatmap
disturbance_df_copy['Month'] = pd.to_datetime(disturbance_df_copy[datetime_col]).dt.month
disturbance_df_copy['Year'] = pd.to_datetime(disturbance_df_copy[datetime_col]).dt.year

heatmap_data2 = disturbance_df_copy.groupby(['Year', 'Month']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(heatmap_data2, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Number of Disturbances'})
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Year', fontsize=12)
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
ax.set_title('Disturbance Heatmap: Month vs Year', fontsize=14, fontweight='bold')
save_figure(fig, '15_heatmap_month_year')

# ============================================================
# 8. VOLTAGE ANALYSIS
# ============================================================
print("\n8️⃣  Voltage Analysis Figures...")

if 'Voltage' in pmu_df.columns:
    voltage_dist = merged_df.groupby('Voltage').size().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    voltage_dist.head(15).plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
    ax.set_xlabel('Voltage Level', fontsize=12)
    ax.set_ylabel('Number of Disturbances', fontsize=12)
    ax.set_title('Disturbances by Voltage Level', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, '16_voltage_distribution')

# ============================================================
# 9. FORECASTING VISUALIZATION
# ============================================================
print("\n9️⃣  Forecasting Figures...")

# Show recent data with forecast
forecast_result = predictive.forecast_arima(ts, order=(1,1,1), forecast_periods=90)

fig, ax = plt.subplots(figsize=(14, 6))
# Plot last 365 days
recent_ts = ts.tail(365)
ax.plot(recent_ts.index, recent_ts.values, label='Historical', color='steelblue', alpha=0.7)
ax.plot(forecast_result['forecast'].index, forecast_result['forecast'].values, 
        label='90-day Forecast', color='red', linewidth=2)

# Confidence interval
ci = forecast_result['confidence_interval']
ax.fill_between(forecast_result['forecast'].index, 
                ci.iloc[:, 0], ci.iloc[:, 1], 
                color='red', alpha=0.2, label='95% CI')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Disturbances', fontsize=12)
ax.set_title('90-Day Disturbance Forecast with Confidence Interval', fontsize=14, fontweight='bold')
ax.legend()
save_figure(fig, '17_forecast_90days')

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("✅ ALL VISUALIZATIONS GENERATED!")
print(f"{'='*60}")
print(f"\nFigures saved to: {FIGURE_DIR}")
print(f"Total figures: 17")

# List all figures
print("\nGenerated figures:")
for f in sorted(os.listdir(FIGURE_DIR)):
    if f.endswith('.png'):
        print(f"  - {f}")
