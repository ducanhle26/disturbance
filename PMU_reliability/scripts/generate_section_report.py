#!/usr/bin/env python3
"""
Section-Specific Report Generator.

Generates comprehensive analysis report for a specific PMU section.

Usage:
    python scripts/generate_section_report.py --section_id 150
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from datetime import datetime

from data_loader import load_pmu_disturbance_data, get_section_events, calculate_event_statistics, get_network_statistics
from risk_scorer import PMURiskScorer
from temporal_analysis import TemporalAnalyzer
from spatial_analysis import SpatialAnalyzer
from visualization import create_section_report_figures

DATA_FILE = Path(__file__).parent.parent.parent / 'data' / 'PMU_disturbance.xlsx'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'


def generate_report(section_id: int):
    print("=" * 60)
    print(f"PMU Reliability Framework - Section {section_id} Report")
    print("=" * 60)
    
    print("\n[1/6] Loading data...")
    pmu_df, dist_df = load_pmu_disturbance_data(str(DATA_FILE))
    
    print(f"\n[2/6] Extracting Section {section_id} events...")
    section_events = get_section_events(dist_df, section_id)
    
    if len(section_events) == 0:
        print(f"ERROR: No events found for Section {section_id}")
        return None
    
    stats = calculate_event_statistics(section_events)
    network_stats = get_network_statistics(dist_df, pmu_df)
    
    print(f"  - Found {stats['count']} events")
    print(f"  - MTBF: {stats['mtbf_days']:.2f} days" if stats['mtbf_days'] else "  - MTBF: N/A")
    
    print(f"\n[3/6] Calculating risk scores...")
    scorer = PMURiskScorer(pmu_df, dist_df)
    risk_results = scorer.calculate_risk_scores()
    section_risk = scorer.get_section_risk(section_id)
    
    print(f"  - Risk Score: {section_risk['risk_score']:.1f}")
    print(f"  - Rank: #{int(section_risk['rank'])} of {len(risk_results)}")
    print(f"  - Category: {section_risk['category']}")
    
    print(f"\n[4/6] Analyzing temporal patterns...")
    analyzer = TemporalAnalyzer(section_events)
    peaks = analyzer.calculate_peak_periods()
    clustering = analyzer.test_clustering()
    
    print(f"  - Peak Hour: {peaks['peak_hour']}:00")
    print(f"  - Peak Day: {peaks['peak_day']}")
    print(f"  - Clustering: {clustering['interpretation']}")
    
    print(f"\n[5/6] Finding similar sections...")
    spatial = SpatialAnalyzer(pmu_df, dist_df)
    similar = spatial.find_similar_sections(section_id, k=10)
    
    print(f"  - Found {len(similar)} similar sections")
    
    print(f"\n[6/6] Generating visualizations...")
    figures_dir = OUTPUT_DIR / 'figures' / 'section_reports' / f'section_{section_id}'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    created_figures = create_section_report_figures(
        section_events, section_id, str(figures_dir),
        risk_results=risk_results, similar_sections=similar
    )
    print(f"  - Created {len(created_figures)} figures")
    
    print(f"\n[7/7] Writing report...")
    reports_dir = OUTPUT_DIR / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f'section_{section_id}_report.md'
    
    network_avg = network_stats['mean_events_per_section']
    ratio = stats['count'] / network_avg if network_avg > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write(f"# Section {section_id} - Deep Dive Analysis Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"Section {section_id} is ranked **#{int(section_risk['rank'])}** out of ")
        f.write(f"{len(risk_results)} sections with a risk score of **{section_risk['risk_score']:.1f}** ")
        f.write(f"({section_risk['category']} Risk).\n\n")
        
        f.write("## Key Metrics\n\n")
        f.write("| Metric | Value | Network Average |\n")
        f.write("|--------|-------|----------------|\n")
        f.write(f"| Total Events | {stats['count']} | {network_avg:.1f} |\n")
        f.write(f"| Event Ratio | {ratio:.1f}x | 1.0x |\n")
        mtbf_str = f"{stats['mtbf_days']:.1f} days" if stats['mtbf_days'] else "N/A"
        f.write(f"| MTBF | {mtbf_str} | - |\n")
        f.write(f"| Risk Score | {section_risk['risk_score']:.1f} | 50 |\n")
        f.write(f"| Risk Rank | #{int(section_risk['rank'])} | - |\n\n")
        
        f.write("## Temporal Patterns\n\n")
        f.write(f"- **Peak Hour**: {peaks['peak_hour']}:00 ({peaks['peak_hour_count']} events)\n")
        f.write(f"- **Peak Day**: {peaks['peak_day']} ({peaks['peak_day_count']} events)\n")
        f.write(f"- **Peak Month**: {peaks['peak_month']} ({peaks['peak_month_count']} events)\n")
        f.write(f"- **Clustering**: {clustering['interpretation']}\n")
        if clustering['dispersion_index']:
            f.write(f"- **Dispersion Index**: {clustering['dispersion_index']:.2f}\n")
        f.write("\n")
        
        f.write("## Similar Sections Comparison\n\n")
        if len(similar) > 0:
            f.write("| Section | Events | Similarity |\n")
            f.write("|---------|--------|------------|\n")
            for _, row in similar.head(5).iterrows():
                f.write(f"| {int(row['SectionID'])} | {int(row['Event_Count'])} | {row['similarity']:.2f} |\n")
            
            best_practice = similar.iloc[0] if len(similar) > 0 else None
            if best_practice is not None and best_practice['Event_Count'] < stats['count']:
                f.write(f"\n**Best Practice**: Section {int(best_practice['SectionID'])} has only ")
                f.write(f"{int(best_practice['Event_Count'])} events with similar characteristics.\n")
        f.write("\n")
        
        f.write("## Risk Score Components\n\n")
        f.write("| Component | Score | Weight |\n")
        f.write("|-----------|-------|--------|\n")
        f.write(f"| Frequency | {section_risk.get('frequency_score', 0):.1f} | 35% |\n")
        f.write(f"| Trend | {section_risk.get('trend_score', 0):.1f} | 25% |\n")
        f.write(f"| MTBF | {section_risk.get('mtbf_score', 0):.1f} | 20% |\n")
        f.write(f"| Age | {section_risk.get('age_score', 0):.1f} | 10% |\n")
        f.write(f"| Recency | {section_risk.get('recency_score', 0):.1f} | 10% |\n")
        f.write(f"| **Total** | **{section_risk['risk_score']:.1f}** | 100% |\n\n")
        
        f.write("## Generated Figures\n\n")
        for fig_path in created_figures:
            fig_name = Path(fig_path).name
            f.write(f"- [{fig_name}](figures/section_reports/section_{section_id}/{fig_name})\n")
        f.write("\n")
        
        f.write("---\n")
        f.write(f"*Report generated by PMU Reliability Framework v1.0*\n")
    
    print(f"  - Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("Report Generation Complete!")
    print("=" * 60)
    
    return {
        'section_id': section_id,
        'stats': stats,
        'risk': section_risk,
        'peaks': peaks,
        'clustering': clustering,
        'similar_sections': similar,
        'figures': created_figures,
        'report_path': str(report_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Generate section-specific analysis report')
    parser.add_argument('--section_id', type=int, required=True, help='Section ID to analyze')
    args = parser.parse_args()
    
    generate_report(args.section_id)


if __name__ == '__main__':
    main()
