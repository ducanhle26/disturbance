#!/usr/bin/env python3
"""
Full Network Analysis Script.

Analyzes all 533 sections, calculates risk scores, and generates summary reports.

Usage:
    python scripts/run_full_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

from data_loader import load_pmu_disturbance_data, get_network_statistics
from risk_scorer import PMURiskScorer

DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'PMU_disturbance.xlsx'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'


def main():
    print("=" * 60)
    print("PMU Reliability Framework - Full Network Analysis")
    print("=" * 60)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[1/5] Loading data...")
    pmu_df, dist_df = load_pmu_disturbance_data(str(DATA_FILE))
    print(f"  - Loaded {len(pmu_df)} PMU sections")
    print(f"  - Loaded {len(dist_df)} disturbance events")
    
    print("\n[2/5] Computing network statistics...")
    network_stats = get_network_statistics(dist_df, pmu_df)
    print(f"  - Total sections with events: {network_stats['total_sections']}")
    print(f"  - Mean events per section: {network_stats['mean_events_per_section']:.2f}")
    print(f"  - Max events per section: {network_stats['max_events_per_section']}")
    
    print("\n[3/5] Calculating risk scores for all sections...")
    scorer = PMURiskScorer(pmu_df, dist_df)
    risk_results = scorer.calculate_risk_scores()
    print(f"  - Scored {len(risk_results)} sections")
    
    category_counts = risk_results['category'].value_counts()
    print(f"  - High risk: {category_counts.get('High', 0)} sections")
    print(f"  - Medium risk: {category_counts.get('Medium', 0)} sections")
    print(f"  - Low risk: {category_counts.get('Low', 0)} sections")
    
    print("\n[4/5] Saving results...")
    results_dir = OUTPUT_DIR / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_scores_path = results_dir / 'network_risk_scores.csv'
    risk_results.to_csv(all_scores_path, index=False)
    print(f"  - Saved all scores to: {all_scores_path}")
    
    top_20 = risk_results.head(20)
    top_scores_path = results_dir / 'top_risk_sections.csv'
    top_20.to_csv(top_scores_path, index=False)
    print(f"  - Saved top 20 to: {top_scores_path}")
    
    print("\n[5/5] Generating summary report...")
    reports_dir = OUTPUT_DIR / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = reports_dir / 'network_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PMU RELIABILITY FRAMEWORK - NETWORK ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATA OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total PMU Sections: {len(pmu_df)}\n")
        f.write(f"Total Disturbance Events: {len(dist_df)}\n")
        f.write(f"Sections with Events: {network_stats['total_sections']}\n")
        if network_stats['time_range']:
            f.write(f"Time Range: {network_stats['time_range'][0]} to {network_stats['time_range'][1]}\n")
        f.write("\n")
        
        f.write("NETWORK STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Events per Section: {network_stats['mean_events_per_section']:.2f}\n")
        f.write(f"Median Events per Section: {network_stats['median_events_per_section']:.2f}\n")
        f.write(f"Std Dev: {network_stats['std_events_per_section']:.2f}\n")
        f.write(f"Max Events: {network_stats['max_events_per_section']}\n")
        f.write(f"Min Events: {network_stats['min_events_per_section']}\n")
        f.write("\n")
        
        f.write("RISK DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"High Risk Sections: {category_counts.get('High', 0)}\n")
        f.write(f"Medium Risk Sections: {category_counts.get('Medium', 0)}\n")
        f.write(f"Low Risk Sections: {category_counts.get('Low', 0)}\n")
        f.write("\n")
        
        f.write("TOP 20 HIGHEST RISK SECTIONS\n")
        f.write("-" * 40 + "\n")
        for _, row in top_20.iterrows():
            f.write(f"Rank {int(row['rank']):2d}: Section {int(row['SectionID']):4d} "
                   f"(Score: {row['risk_score']:.1f}, Events: {int(row['event_count'])})\n")
    
    print(f"  - Saved summary to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nTop 5 Highest Risk Sections:")
    for _, row in risk_results.head(5).iterrows():
        print(f"  #{int(row['rank'])}: Section {int(row['SectionID'])} "
              f"(Score: {row['risk_score']:.1f}, Events: {int(row['event_count'])})")
    
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    return risk_results


if __name__ == '__main__':
    main()
