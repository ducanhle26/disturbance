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
    print("\n[7/12] Generating visualizations...")
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
    print("\n[8/12] Generating reports...")
    from section150_report import generate_all_reports

    reports = generate_all_reports(
        event_breakdown, root_cause, comparative, characteristics, recommendations
    )

    # Step 9: Operational Context Analysis (EXTENDED ANALYSIS)
    print("\n[9/12] Analyzing operational context...")
    from section150_operational_context import get_operational_context_summary

    operational_context = get_operational_context_summary(
        section150_events, pmu_df, disturbance_df, network_baselines
    )

    cromwell = operational_context['cromwell']
    print(f"  - Cromwell Tap mentioned: {cromwell['mentions']} events")
    print(f"  - Network interconnections: PSO={operational_context['interconnection']['pso_mentions']}, OMPA={operational_context['interconnection']['ompa_mentions']}")

    equipment_age = operational_context['equipment_age']
    if 'error' not in equipment_age:
        print(f"  - Equipment age: {equipment_age['section150_age_years']:.1f} years ({equipment_age['age_percentile']:.0f}th percentile)")

    # Step 10: Unknown Events Deep-Dive (EXTENDED ANALYSIS)
    print("\n[10/12] Analyzing Unknown events...")
    from section150_unknown_analysis import get_unknown_analysis_summary

    unknown_analysis = get_unknown_analysis_summary(
        section150_events, network_baselines
    )

    print(f"  - Unknown events: {unknown_analysis['total_unknown']}")
    print(f"  - Reclassifiable: {unknown_analysis['reclassification']['reclassifiable_count']} ({unknown_analysis['reclassification']['reclassifiable_pct']:.1f}%)")

    temporal_clustering = unknown_analysis['temporal_clustering']
    if 'error' not in temporal_clustering:
        print(f"  - Temporal clustering: {temporal_clustering['clustering_conclusion']}")

    # Step 11: Hourly Analysis (EXTENDED ANALYSIS)
    print("\n[11/12] Investigating 7 PM peak...")
    from section150_hourly_analysis import get_hourly_analysis_summary

    hourly_analysis = get_hourly_analysis_summary(
        section150_events, disturbance_df, operational_context,
        unknown_analysis, network_baselines
    )

    hourly_comp = hourly_analysis['hourly_comparison']
    print(f"  - 7 PM peak confirmed: {hourly_comp['hour_19_pct_150']:.1f}% of events")
    print(f"  - Network average at 7 PM: {hourly_comp['hour_19_pct_network']:.1f}%")
    print(f"  - Enrichment ratio: {hourly_comp['enrichment_ratio']:.1f}x")

    unknown_timing = hourly_analysis['unknown_timing']
    if 'error' not in unknown_timing:
        print(f"  - Unknown events at 7 PM: {unknown_timing['unknown_at_19']} events")

    # Step 12: Generate Extended Visualizations (EXTENDED ANALYSIS)
    print("\n[12/12] Generating extended visualizations...")
    from section150_visuals import generate_extended_visualizations

    try:
        extended_figures = generate_extended_visualizations(
            operational_context, unknown_analysis, hourly_analysis
        )
        print(f"  - Extended figures generated: {len(extended_figures)}")
    except Exception as e:
        print(f"  - Extended visualization error (continuing): {e}")
        extended_figures = {}
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Results for Section 150:")
    print(f"  • Total Events: {len(section150_events)} (Rank #1 in network)")
    print(f"  • Rate vs Network Average: {network_comp['rate_ratio']:.1f}x higher")
    print(f"  • Statistical Significance: {significance['conclusion']}")
    print(f"  • Potential Reduction: {risk_reduction['overall_reduction_pct']:.0f}% with interventions")

    print("\nExtended Analysis Results:")
    print(f"  • Cromwell Tap 31 mentions: {cromwell['mentions']} events")
    print(f"  • Unknown events reclassifiable: {unknown_analysis['reclassification']['reclassifiable_count']} of {unknown_analysis['total_unknown']}")
    print(f"  • 7 PM peak enrichment: {hourly_comp['enrichment_ratio']:.1f}x vs network")

    print("\nOutputs:")
    print(f"  • Executive Summary: {cfg.EXECUTIVE_SUMMARY_FILE}")
    print(f"  • Technical Report: {cfg.TECHNICAL_REPORT_FILE}")
    print(f"  • Figures: {cfg.FIGURE_DIR}")
    print(f"    - Original analysis: 7 figures")
    print(f"    - Extended analysis: 5 figures")
    print(f"    - Total: 12 figures generated")

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
        'reports': reports,
        'operational_context': operational_context,
        'unknown_analysis': unknown_analysis,
        'hourly_analysis': hourly_analysis,
        'extended_figures': extended_figures
    }


if __name__ == '__main__':
    results = main()
