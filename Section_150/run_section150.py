#!/usr/bin/env python3
"""
Main runner for Section 150 deep-dive analysis.
Executes all analysis modules and generates outputs.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'EDA'))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'EDA', 'src'))

import config_section150 as cfg


def main():
    """Run complete Section 150 analysis."""
    print("=" * 70)
    print("SECTION 150 DEEP-DIVE ANALYSIS")
    print("=" * 70)
    print(f"Target Section: {cfg.TARGET_SECTION_ID}")
    print(f"Output Directory: {cfg.OUTPUT_DIR}")
    print("=" * 70)
    
    # Step 1: Load Data
    print("\n[1/7] Loading data...")
    from section150_loader import (
        load_section150_data,
        compute_network_baselines,
        get_section150_pmu_info
    )
    
    section150_events, disturbance_df, pmu_df, merged_df = load_section150_data(use_cache=False)
    network_baselines = compute_network_baselines(disturbance_df, pmu_df)
    
    print(f"  - Section 150 events: {len(section150_events)}")
    print(f"  - Total network events: {network_baselines['total_events']}")
    print(f"  - Network average: {network_baselines['mean_events_per_section']:.1f} events/section")
    print(f"  - Datetime column: {network_baselines['datetime_col']}")
    print(f"  - Cause column: {network_baselines['cause_col']}")
    
    # Step 2: Event Breakdown
    print("\n[2/7] Analyzing event breakdown...")
    from section150_event_breakdown import get_event_breakdown_summary
    
    event_breakdown = get_event_breakdown_summary(
        section150_events, disturbance_df, network_baselines
    )
    
    clustering = event_breakdown['clustering']
    print(f"  - Temporal clustering: {clustering['clustering_conclusion']}")
    print(f"  - Burst days detected: {clustering['n_burst_days']}")
    print(f"  - Change points: {len(clustering['change_points'])}")
    
    network_comp = event_breakdown['network_comparison']
    print(f"  - Rate ratio vs network: {network_comp['rate_ratio']:.1f}x")
    print(f"  - Percentile rank: {network_comp['percentile_rank']:.1f}%")
    
    # Step 3: Section Characteristics
    print("\n[3/7] Analyzing section characteristics...")
    from section150_characteristics import get_characteristics_summary
    
    characteristics = get_characteristics_summary(pmu_df, disturbance_df)
    
    sec150_details = characteristics['section150_details']
    print(f"  - PMU details extracted: {len(sec150_details)} attributes")
    print(f"  - Similar sections found: {len(characteristics['similar_sections'])}")
    print(f"  - Unique features identified: {len(characteristics['unique_features'])}")
    
    # Step 4: Root Cause Analysis
    print("\n[4/7] Performing root cause analysis...")
    from section150_root_cause import get_root_cause_summary
    
    root_cause = get_root_cause_summary(
        section150_events, disturbance_df, network_baselines
    )
    
    significance = root_cause['significance_tests']
    print(f"  - Failure rate significance: {significance['conclusion']}")
    print(f"  - Z-score: {significance['z_score']:.2f}")
    
    seasonal = root_cause['seasonal_patterns']
    print(f"  - Peak hour: {seasonal['peak_hour']}:00")
    print(f"  - Peak month: {seasonal['peak_month']}")
    print(f"  - Seasonal variation: {seasonal['seasonal_variation']}")
    
    # Step 5: Comparative Analysis
    print("\n[5/7] Performing comparative analysis...")
    from section150_comparative import get_comparative_summary
    
    comparative = get_comparative_summary(
        section150_events, disturbance_df, pmu_df, network_baselines
    )
    
    print(f"  - Similar sections analyzed: {comparative['n_similar_analyzed']}")
    print(f"  - Key learnings: {len(comparative['learnings'])}")
    
    # Step 6: Generate Recommendations
    print("\n[6/7] Generating recommendations...")
    from section150_recommendations import get_recommendations_summary
    
    recommendations = get_recommendations_summary(
        event_breakdown, root_cause, comparative, characteristics
    )
    
    risk_reduction = recommendations['risk_reduction_estimate']
    print(f"  - Potential event reduction: {risk_reduction['overall_reduction_pct']:.0f}%")
    print(f"  - Events preventable: {risk_reduction['total_preventable']:.0f}")
    print(f"  - Priority actions: {len(recommendations['key_recommendations'])}")
    
    # Step 7: Generate Visualizations
    print("\n[7/7] Generating visualizations...")
    from section150_visuals import generate_all_visualizations
    
    try:
        figures = generate_all_visualizations(
            event_breakdown, root_cause, comparative, characteristics
        )
        print(f"  - Figures generated: {len(figures)}")
    except Exception as e:
        print(f"  - Visualization error (continuing): {e}")
        figures = {}
    
    # Step 8: Generate Reports
    print("\n[8/7] Generating reports...")
    from section150_report import generate_all_reports
    
    reports = generate_all_reports(
        event_breakdown, root_cause, comparative, characteristics, recommendations
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Results for Section 150:")
    print(f"  • Total Events: {len(section150_events)} (Rank #1 in network)")
    print(f"  • Rate vs Network Average: {network_comp['rate_ratio']:.1f}x higher")
    print(f"  • Statistical Significance: {significance['conclusion']}")
    print(f"  • Potential Reduction: {risk_reduction['overall_reduction_pct']:.0f}% with interventions")
    
    print("\nOutputs:")
    print(f"  • Executive Summary: {cfg.EXECUTIVE_SUMMARY_FILE}")
    print(f"  • Technical Report: {cfg.TECHNICAL_REPORT_FILE}")
    print(f"  • Figures: {cfg.FIGURE_DIR}")
    
    print("\nTop 3 Recommendations:")
    for rec in recommendations['key_recommendations'][:3]:
        print(f"  {rec['priority']}. {rec['recommendation']}")
    
    print("\n" + "=" * 70)
    
    return {
        'event_breakdown': event_breakdown,
        'characteristics': characteristics,
        'root_cause': root_cause,
        'comparative': comparative,
        'recommendations': recommendations,
        'figures': figures,
        'reports': reports
    }


if __name__ == '__main__':
    results = main()
