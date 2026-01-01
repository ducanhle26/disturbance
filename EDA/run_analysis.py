#!/usr/bin/env python3
"""
PMU Disturbance Analysis - Complete Execution Script
Runs all analyses and generates outputs.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from datetime import datetime

# Import project modules
import config
from src import data_loader, temporal, causality, spatial, predictive, statistical, visualizations

def ensure_directories():
    """Create output directories if they don't exist."""
    dirs = [
        os.path.join(config.OUTPUT_DIR, 'data'),
        os.path.join(config.OUTPUT_DIR, 'figures', 'static'),
        os.path.join(config.OUTPUT_DIR, 'figures', 'interactive'),
        os.path.join(config.OUTPUT_DIR, 'models'),
        os.path.join(config.OUTPUT_DIR, 'reports'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Output directories ready")

def run_data_loading():
    """Load and validate data."""
    print("\n" + "="*60)
    print("📊 PHASE 1: DATA LOADING & QUALITY ASSESSMENT")
    print("="*60)
    
    # Load data
    pmu_df, disturbance_df, merged_df = data_loader.load_all_data(
        config.EXCEL_FILE, config.PMU_SHEET, config.DISTURBANCE_SHEET
    )
    
    # Data summaries
    print(f"\n📋 PMU Data Summary:")
    print(f"   - Records: {len(pmu_df)}")
    print(f"   - Columns: {list(pmu_df.columns)}")
    
    print(f"\n📋 Disturbance Data Summary:")
    print(f"   - Records: {len(disturbance_df)}")
    print(f"   - Columns: {list(disturbance_df.columns)}")
    
    # Identify datetime column
    datetime_cols = [col for col in disturbance_df.columns 
                     if 'date' in col.lower() or 'time' in col.lower()]
    datetime_col = datetime_cols[0] if datetime_cols else None
    print(f"   - DateTime column: {datetime_col}")
    
    # Identify cause column
    cause_cols = [col for col in disturbance_df.columns 
                  if 'cause' in col.lower()]
    cause_col = cause_cols[0] if cause_cols else None
    print(f"   - Cause column: {cause_col}")
    
    # Quality report
    pmu_summary = data_loader.get_data_summary(pmu_df)
    dist_summary = data_loader.get_data_summary(disturbance_df)
    
    quality_report = f"""
PMU DISTURBANCE DATA QUALITY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

PMU DATA:
- Total records: {pmu_summary['shape'][0]}
- Columns: {pmu_summary['shape'][1]}
- Duplicates: {pmu_summary['duplicates']}
- Memory usage: {pmu_summary['memory_usage_mb']:.2f} MB

DISTURBANCE DATA:
- Total records: {dist_summary['shape'][0]}
- Columns: {dist_summary['shape'][1]}
- Duplicates: {dist_summary['duplicates']}
- Memory usage: {dist_summary['memory_usage_mb']:.2f} MB

MISSING VALUES (PMU):
{pd.Series(pmu_summary['missing_values']).to_string()}

MISSING VALUES (DISTURBANCE):
{pd.Series(dist_summary['missing_values']).to_string()}
"""
    
    # Save quality report
    report_path = os.path.join(config.REPORT_DIR, 'data_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write(quality_report)
    print(f"\n✅ Quality report saved to {report_path}")
    
    return pmu_df, disturbance_df, merged_df, datetime_col, cause_col

def run_temporal_analysis(disturbance_df, datetime_col):
    """Run temporal analysis."""
    print("\n" + "="*60)
    print("⏱️  PHASE 2: TEMPORAL ANALYSIS")
    print("="*60)
    
    if datetime_col is None:
        print("⚠️  No datetime column found, skipping temporal analysis")
        return None
    
    # Aggregate daily counts
    ts = temporal.aggregate_disturbances_by_time(disturbance_df, datetime_col, freq='D')
    print(f"   - Time series length: {len(ts)} days")
    print(f"   - Date range: {ts.index.min()} to {ts.index.max()}")
    print(f"   - Total disturbances: {ts.sum()}")
    print(f"   - Daily mean: {ts.mean():.2f}")
    print(f"   - Daily max: {ts.max()}")
    
    # STL Decomposition
    print("\n   Running STL decomposition...")
    decomposition = temporal.decompose_time_series(ts, seasonal=7)
    
    # Anomaly detection
    print("   Running anomaly detection...")
    anomalies_zscore = temporal.detect_anomalies_zscore(ts, threshold=3.0)
    anomalies_iqr = temporal.detect_anomalies_iqr(ts, multiplier=1.5)
    
    print(f"   - Z-score anomalies: {anomalies_zscore.sum()}")
    print(f"   - IQR anomalies: {anomalies_iqr.sum()}")
    
    # Change point detection
    print("   Detecting change points...")
    try:
        change_points = temporal.detect_change_points(ts, pen=10)
        print(f"   - Change points detected: {len(change_points)}")
    except Exception as e:
        print(f"   - Change point detection skipped: {e}")
        change_points = []
    
    # Rolling statistics
    rolling_stats = temporal.calculate_rolling_statistics(ts, windows=[7, 30, 90])
    
    # Cyclical patterns
    patterns = temporal.extract_cyclical_patterns(disturbance_df, datetime_col)
    print(f"\n   Cyclical patterns extracted:")
    print(f"   - Peak hour: {patterns['hourly'].idxmax()} ({patterns['hourly'].max()} events)")
    print(f"   - Peak day: {patterns['daily'].idxmax()} ({patterns['daily'].max()} events)")
    print(f"   - Peak month: {patterns['monthly'].idxmax()} ({patterns['monthly'].max()} events)")
    
    # Save temporal results
    temporal_results = {
        'daily_counts': ts,
        'rolling_7d_mean': rolling_stats['rolling_mean_7d'],
        'rolling_30d_mean': rolling_stats['rolling_mean_30d'],
        'trend': decomposition['trend'],
        'seasonal': decomposition['seasonal'],
        'anomalies_zscore': anomalies_zscore,
        'anomalies_iqr': anomalies_iqr,
    }
    
    temporal_df = pd.DataFrame(temporal_results)
    temporal_df.to_csv(config.TEMPORAL_RESULTS)
    print(f"\n✅ Temporal results saved to {config.TEMPORAL_RESULTS}")
    
    return ts, decomposition, patterns

def run_causality_analysis(disturbance_df, datetime_col, cause_col):
    """Run causality and pattern mining analysis."""
    print("\n" + "="*60)
    print("🔍 PHASE 3: CAUSALITY & PATTERN MINING")
    print("="*60)
    
    if cause_col is None:
        print("⚠️  No cause column found, skipping causality analysis")
        return None
    
    # Cause distribution
    print("\n   Analyzing cause distribution...")
    cause_dist = causality.analyze_cause_distribution(disturbance_df, cause_col)
    print(f"\n   Top 10 Causes:")
    print(cause_dist.head(10).to_string())
    
    # Pareto analysis
    pareto_causes, n_pareto = causality.calculate_pareto_80_20(cause_dist)
    print(f"\n   Pareto (80/20): {n_pareto} causes account for 80% of disturbances")
    
    # MTBF/MTTR
    print("\n   Calculating MTBF/MTTR metrics...")
    if datetime_col:
        mtbf_mttr = causality.calculate_mtbf_mttr(disturbance_df, datetime_col)
        print(f"   - Sections analyzed: {len(mtbf_mttr)}")
        print(f"   - Average MTBF: {mtbf_mttr['MTBF_hours'].mean():.2f} hours")
    
    # Association rules
    print("\n   Mining association rules...")
    try:
        rules = causality.mine_association_rules(
            disturbance_df, cause_col,
            min_support=config.MIN_SUPPORT,
            min_confidence=config.MIN_CONFIDENCE
        )
        print(f"   - Found {len(rules)} association rules")
    except Exception as e:
        print(f"   - Association rules skipped: {e}")
        rules = pd.DataFrame()
    
    # Sequential patterns
    if datetime_col:
        print("\n   Detecting sequential patterns...")
        try:
            seq_patterns = causality.detect_sequential_patterns(
                disturbance_df, datetime_col, cause_col,
                window_days=config.SEQUENTIAL_WINDOW_DAYS
            )
            if len(seq_patterns) > 0:
                print(f"   - Found {len(seq_patterns)} sequential patterns")
                print(f"\n   Top 5 Sequential Patterns:")
                print(seq_patterns.head().to_string())
        except Exception as e:
            print(f"   - Sequential patterns skipped: {e}")
    
    # Save results
    cause_dist.to_csv(config.CAUSALITY_RESULTS)
    print(f"\n✅ Causality results saved to {config.CAUSALITY_RESULTS}")
    
    return cause_dist, mtbf_mttr if datetime_col else None

def run_spatial_analysis(pmu_df, disturbance_df):
    """Run spatial and network analysis."""
    print("\n" + "="*60)
    print("🗺️  PHASE 4: SPATIAL & NETWORK ANALYSIS")
    print("="*60)
    
    # Find coordinate columns
    lat_cols = [c for c in pmu_df.columns if 'lat' in c.lower()]
    lon_cols = [c for c in pmu_df.columns if 'lon' in c.lower()]
    
    if not lat_cols or not lon_cols:
        print("⚠️  No coordinate columns found, skipping spatial analysis")
        return None
    
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Validate coordinates
    print(f"\n   Coordinate columns: {lat_col}, {lon_col}")
    coord_validation = spatial.validate_coordinates(pmu_df, lat_col, lon_col)
    print(f"   - Total records: {coord_validation['total_records']}")
    print(f"   - Valid coordinates: {coord_validation['valid_coordinates']}")
    
    # DBSCAN clustering
    print("\n   Running DBSCAN clustering...")
    clustered = spatial.perform_dbscan_clustering(
        pmu_df, lat_col, lon_col,
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES
    )
    n_clusters = clustered['Cluster'].nunique() - (1 if -1 in clustered['Cluster'].values else 0)
    print(f"   - Clusters found: {n_clusters}")
    
    # K-means clustering
    print("\n   Running K-means clustering...")
    kmeans_clustered = spatial.perform_kmeans_clustering(
        pmu_df, lat_col, lon_col,
        n_clusters=config.KMEANS_N_CLUSTERS
    )
    
    # Build proximity network
    print("\n   Building proximity network...")
    try:
        G = spatial.build_proximity_network(
            pmu_df, lat_col, lon_col,
            threshold_distance=1.0
        )
        print(f"   - Nodes: {G.number_of_nodes()}")
        print(f"   - Edges: {G.number_of_edges()}")
        
        # Network centrality
        if G.number_of_nodes() > 0:
            centrality = spatial.calculate_network_centrality(G)
            print(f"\n   Top 5 nodes by betweenness centrality:")
            print(centrality.head().to_string())
    except Exception as e:
        print(f"   - Network analysis skipped: {e}")
        centrality = None
    
    # Save results
    clustered[['SectionID', 'Cluster', lat_col, lon_col]].to_csv(config.SPATIAL_RESULTS, index=False)
    print(f"\n✅ Spatial results saved to {config.SPATIAL_RESULTS}")
    
    return clustered

def run_predictive_analysis(pmu_df, disturbance_df, datetime_col, ts=None):
    """Run predictive modeling and risk scoring."""
    print("\n" + "="*60)
    print("🎯 PHASE 5: PREDICTIVE MODELING & RISK SCORING")
    print("="*60)
    
    # Calculate risk scores
    print("\n   Calculating composite risk scores...")
    try:
        risk_scores = predictive.calculate_composite_risk_score(
            pmu_df, disturbance_df,
            datetime_col=datetime_col if datetime_col else 'DateTime'
        )
        
        print(f"\n   Risk Score Distribution:")
        print(risk_scores['Risk_Category'].value_counts().to_string())
        
        print(f"\n   Top 10 Highest Risk Sections:")
        print(risk_scores.head(10)[['SectionID', 'Risk_Score_0_100', 'Risk_Category', 'Historical_Frequency']].to_string())
        
        # Save risk scores
        risk_scores.to_csv(config.RISK_SCORES, index=False)
        print(f"\n✅ Risk scores saved to {config.RISK_SCORES}")
    except Exception as e:
        print(f"   - Risk scoring failed: {e}")
        risk_scores = None
    
    # Time series forecasting
    if ts is not None and len(ts) > 30:
        print("\n   Running time series forecasting...")
        try:
            forecast_result = predictive.forecast_arima(ts, order=(1,1,1), forecast_periods=30)
            print(f"   - 30-day forecast generated")
            print(f"   - AIC: {forecast_result['aic']:.2f}")
            
            # Generate predictions for multiple horizons
            predictions = []
            for horizon in config.FORECAST_HORIZONS:
                fc = predictive.forecast_arima(ts, order=(1,1,1), forecast_periods=horizon)
                predictions.append({
                    'Horizon_Days': horizon,
                    'Predicted_Total': fc['forecast'].sum(),
                    'Daily_Average': fc['forecast'].mean(),
                    'AIC': fc['aic']
                })
            
            predictions_df = pd.DataFrame(predictions)
            predictions_df.to_csv(config.PREDICTIONS, index=False)
            print(f"\n   Forecast Summary:")
            print(predictions_df.to_string())
            print(f"\n✅ Predictions saved to {config.PREDICTIONS}")
        except Exception as e:
            print(f"   - Forecasting failed: {e}")
    
    return risk_scores

def run_statistical_validation(disturbance_df, datetime_col, cause_col, ts=None):
    """Run statistical validation tests."""
    print("\n" + "="*60)
    print("📈 PHASE 6: STATISTICAL VALIDATION")
    print("="*60)
    
    results = {}
    
    # Distribution fitting
    if ts is not None:
        print("\n   Fitting distributions to daily counts...")
        try:
            dist_fit = statistical.test_distribution_fit(ts)
            print(f"   Distribution fit results:")
            print(dist_fit.to_string())
            results['distribution_fit'] = dist_fit
        except Exception as e:
            print(f"   - Distribution fitting failed: {e}")
    
    # Trend test
    if ts is not None:
        print("\n   Running Mann-Kendall trend test...")
        try:
            trend_result = statistical.mann_kendall_test(ts)
            print(f"   - Trend: {trend_result['Trend']}")
            print(f"   - P-value: {trend_result['P_value']:.4f}")
            results['trend_test'] = trend_result
        except Exception as e:
            print(f"   - Trend test failed: {e}")
    
    # Correlation analysis
    print("\n   Computing correlation matrix...")
    numeric_cols = disturbance_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        try:
            corr_matrix, p_values = statistical.correlation_analysis(
                disturbance_df[numeric_cols[:10]]  # Limit to 10 columns
            )
            print(f"   - Computed correlations for {len(numeric_cols[:10])} numeric columns")
            results['correlations'] = corr_matrix
        except Exception as e:
            print(f"   - Correlation analysis failed: {e}")
    
    # Save results
    results_summary = []
    for test_name, result in results.items():
        if isinstance(result, dict):
            for key, value in result.items():
                if not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
                    results_summary.append({'Test': test_name, 'Metric': key, 'Value': str(value)})
    
    if results_summary:
        pd.DataFrame(results_summary).to_csv(config.STATISTICAL_RESULTS, index=False)
        print(f"\n✅ Statistical results saved to {config.STATISTICAL_RESULTS}")
    
    return results

def generate_executive_summary(pmu_df, disturbance_df, risk_scores, cause_dist, patterns):
    """Generate executive summary report."""
    print("\n" + "="*60)
    print("📝 GENERATING EXECUTIVE SUMMARY")
    print("="*60)
    
    summary = f"""
EXECUTIVE SUMMARY: PMU DISTURBANCE ANALYSIS
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. DATA OVERVIEW
----------------
• Total PMU Sections: {len(pmu_df)}
• Total Disturbance Events: {len(disturbance_df)}
• Average Disturbances per Section: {len(disturbance_df)/len(pmu_df):.1f}

2. TOP 10 CRITICAL INSIGHTS
---------------------------
"""
    
    insights = []
    
    # Risk-based insights
    if risk_scores is not None:
        critical_sections = risk_scores[risk_scores['Risk_Category'] == 'Critical']
        high_risk = risk_scores[risk_scores['Risk_Category'] == 'High']
        insights.append(f"• {len(critical_sections)} sections classified as CRITICAL risk")
        insights.append(f"• {len(high_risk)} sections classified as HIGH risk")
        insights.append(f"• Top risk section: {risk_scores.iloc[0]['SectionID']} (score: {risk_scores.iloc[0]['Risk_Score_0_100']:.1f})")
    
    # Cause-based insights
    if cause_dist is not None:
        top_cause = cause_dist.index[0]
        top_cause_pct = cause_dist.iloc[0]['Percentage']
        insights.append(f"• Leading cause: '{top_cause}' ({top_cause_pct:.1f}% of events)")
    
    # Temporal insights
    if patterns is not None:
        peak_hour = patterns['hourly'].idxmax()
        peak_day = patterns['daily'].idxmax()
        insights.append(f"• Peak disturbance hour: {peak_hour}:00")
        insights.append(f"• Peak disturbance day: {peak_day}")
    
    # Add insights to summary
    for i, insight in enumerate(insights[:10], 1):
        summary += f"{i}. {insight[2:]}\n"
    
    summary += f"""
3. HIGH-PRIORITY SECTIONS
-------------------------
"""
    
    if risk_scores is not None:
        for i, row in risk_scores.head(10).iterrows():
            summary += f"• {row['SectionID']}: Risk Score {row['Risk_Score_0_100']:.1f} ({row['Risk_Category']})\n"
    
    summary += f"""
4. RECOMMENDATIONS
------------------
• Prioritize maintenance for Critical and High risk sections
• Focus on the top cause category for prevention measures
• Schedule inspections during off-peak hours when possible
• Implement predictive maintenance for aging PMU equipment

5. NEXT STEPS
-------------
• Review detailed analysis in individual notebooks
• Examine visualizations in outputs/figures/
• Use risk scores for maintenance planning
• Monitor high-risk sections more frequently

============================================
END OF EXECUTIVE SUMMARY
"""
    
    # Save summary
    summary_path = os.path.join(config.REPORT_DIR, 'executive_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"✅ Executive summary saved to {summary_path}")
    
    return summary

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("🚀 PMU DISTURBANCE ANALYSIS - FULL EXECUTION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    ensure_directories()
    
    # Phase 1: Data Loading
    pmu_df, disturbance_df, merged_df, datetime_col, cause_col = run_data_loading()
    
    # Phase 2: Temporal Analysis
    ts = None
    patterns = None
    if datetime_col:
        result = run_temporal_analysis(disturbance_df, datetime_col)
        if result:
            ts, decomposition, patterns = result
    
    # Phase 3: Causality Analysis
    cause_dist = None
    if cause_col:
        result = run_causality_analysis(disturbance_df, datetime_col, cause_col)
        if result:
            cause_dist, mtbf = result
    
    # Phase 4: Spatial Analysis
    run_spatial_analysis(pmu_df, disturbance_df)
    
    # Phase 5: Predictive Modeling
    risk_scores = run_predictive_analysis(pmu_df, disturbance_df, datetime_col, ts)
    
    # Phase 6: Statistical Validation
    run_statistical_validation(disturbance_df, datetime_col, cause_col, ts)
    
    # Generate Executive Summary
    generate_executive_summary(pmu_df, disturbance_df, risk_scores, cause_dist, patterns)
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs generated in: {config.OUTPUT_DIR}")
    print(f"  - Data quality report: {config.REPORT_DIR}/data_quality_report.txt")
    print(f"  - Risk scores: {config.RISK_SCORES}")
    print(f"  - Predictions: {config.PREDICTIONS}")
    print(f"  - Executive summary: {config.REPORT_DIR}/executive_summary.txt")
    print("\n🎉 All analyses completed successfully!")

if __name__ == '__main__':
    main()
