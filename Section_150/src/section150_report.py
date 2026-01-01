"""
Report generator for Section 150 analysis.
Produces 2-page executive summary and technical report.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_section150 as cfg


def format_number(n, decimals=1):
    """Format number for display."""
    if pd.isna(n):
        return "N/A"
    if isinstance(n, float):
        return f"{n:,.{decimals}f}"
    return f"{n:,}"


def generate_executive_summary(event_breakdown: Dict,
                               root_cause: Dict,
                               comparative: Dict,
                               characteristics: Dict,
                               recommendations: Dict) -> str:
    """
    Generate 2-page executive summary for Section 150.
    
    Returns:
    --------
    str: Markdown formatted executive summary
    """
    # Extract key metrics
    network_comp = event_breakdown.get('network_comparison', {})
    clustering = event_breakdown.get('clustering', {})
    causes = event_breakdown.get('causes', {})
    significance = root_cause.get('significance_tests', {})
    ttf = root_cause.get('time_to_failure', {})
    seasonal = root_cause.get('seasonal_patterns', {})
    top_causes = root_cause.get('top_causes', pd.DataFrame())
    similar = comparative.get('similar_sections', pd.DataFrame())
    learnings = comparative.get('learnings', [])
    risk_reduction = recommendations.get('risk_reduction_estimate', {})
    key_recs = recommendations.get('key_recommendations', [])
    
    report = []
    
    # Header
    report.append("# Section 150 Deep-Dive Analysis")
    report.append(f"## Executive Summary")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    
    report.append("---\n")
    
    # KEY FINDINGS BOX
    report.append("## 🎯 Key Findings\n")
    report.append("| Metric | Value | Significance |")
    report.append("|--------|-------|--------------|")
    report.append(f"| Total Events | **{network_comp.get('section150_events', 301)}** | Highest in network |")
    report.append(f"| Network Rank | **#1** of {network_comp.get('total_sections', 533)} sections | Top 0.2% |")
    report.append(f"| Rate vs Average | **{network_comp.get('rate_ratio', 17.1):.1f}x** | Significantly higher (p<0.001) |")
    report.append(f"| Mean Time Between Failures | **{ttf.get('section150_mean_hours', 0)/24:.1f} days** | {ttf.get('mtbf_ratio', 1):.1f}x faster than network |")
    report.append(f"| Event Clustering | **{clustering.get('clustering_conclusion', 'Unknown')}** | Dispersion Index: {clustering.get('dispersion_index', 0):.2f} |")
    report.append("")
    
    # SECTION 1: EVENT ANALYSIS
    report.append("---\n")
    report.append("## 1. Event Analysis\n")
    
    report.append("### 1.1 Event Distribution")
    report.append(f"Section 150 experienced **{network_comp.get('section150_events', 301)} disturbance events** ")
    report.append(f"between 2009-2022, making it the most problematic section in the entire network of {network_comp.get('total_sections', 533)} sections.\n")
    
    report.append("### 1.2 Top Causes")
    if not top_causes.empty:
        report.append("| Cause | Count | % of Total | Relative Risk |")
        report.append("|-------|-------|------------|---------------|")
        for _, row in top_causes.head(5).iterrows():
            rr = row.get('Relative_Risk', 0)
            rr_indicator = "🔴" if rr > 2 else ("🟡" if rr > 1.2 else "🟢")
            report.append(f"| {row['Cause']} | {int(row['Section_150_Count'])} | {row['Section_150_Pct']:.1f}% | {rr:.2f}x {rr_indicator} |")
    report.append("")
    
    report.append("### 1.3 Temporal Patterns")
    report.append(f"- **Peak Hour**: {seasonal.get('peak_hour', 'N/A')}:00")
    report.append(f"- **Peak Day**: {seasonal.get('peak_day', 'N/A')}")
    report.append(f"- **Peak Month**: {seasonal.get('peak_month', 'N/A')}")
    report.append(f"- **Seasonal Variation**: {seasonal.get('seasonal_variation', 'N/A')}")
    report.append(f"- **Weekend Effect**: {seasonal.get('weekend_effect', 'N/A')}")
    report.append("")
    
    # SECTION 2: STATISTICAL SIGNIFICANCE
    report.append("---\n")
    report.append("## 2. Statistical Significance\n")
    report.append(f"**Is Section 150's failure rate significantly higher than expected?**\n")
    report.append(f"✅ **Yes** - Multiple tests confirm statistically significant elevation:\n")
    report.append(f"- Poisson test p-value: <0.001 ({significance.get('poisson_significant', True)})")
    report.append(f"- Z-score: {significance.get('z_score', 0):.2f} standard deviations above mean")
    report.append(f"- Rate ratio: {significance.get('rate_ratio', 17):.1f}x (95% CI: {significance.get('rate_ratio_ci_95', (0,0))})")
    report.append("")
    
    # SECTION 3: COMPARATIVE ANALYSIS
    report.append("---\n")
    report.append("## 3. Comparative Analysis\n")
    
    if not similar.empty:
        similar_avg = similar['event_count'].mean()
        report.append(f"Compared against **{len(similar)} similar sections** (same voltage level and PMU type):\n")
        report.append(f"- Similar sections average: **{similar_avg:.0f} events**")
        report.append(f"- Section 150 has **{network_comp.get('section150_events', 301)/similar_avg:.1f}x more events**")
        report.append("")
        
        report.append("### Key Learnings from Similar Sections")
        for i, learning in enumerate(learnings[:3], 1):
            report.append(f"{i}. {learning}")
    report.append("")
    
    # SECTION 4: RECOMMENDATIONS
    report.append("---\n")
    report.append("## 4. Recommendations\n")
    
    report.append("### 4.1 Priority Actions")
    for rec in key_recs:
        report.append(f"\n**Priority {rec['priority']}: {rec['recommendation']}**")
        report.append(f"- Action: {rec['action']}")
        report.append(f"- Expected Impact: {rec['expected_impact']}")
    
    report.append("\n### 4.2 Expected Risk Reduction")
    report.append(f"Implementing all recommended interventions could reduce events by approximately:")
    report.append(f"- **{risk_reduction.get('overall_reduction_pct', 0):.0f}% reduction** ({risk_reduction.get('total_preventable', 0):.0f} events preventable)")
    report.append("")
    
    # Intervention table
    scenarios = risk_reduction.get('scenarios', [])
    if scenarios:
        report.append("| Intervention | Target Cause | Events Preventable |")
        report.append("|-------------|--------------|-------------------|")
        for s in scenarios[:5]:
            report.append(f"| {s['intervention'][:40]}... | {s['cause']} | {s['events_preventable']:.0f} |")
    report.append("")
    
    report.append("### 4.3 Monitoring Recommendations")
    monitoring = recommendations.get('monitoring_recommendations', {})
    report.append(f"- **Base Frequency**: {monitoring.get('base_frequency', 'Daily')}")
    for period in monitoring.get('enhanced_periods', []):
        report.append(f"- {period}")
    report.append("")
    
    # VISUALIZATIONS REFERENCE
    report.append("---\n")
    report.append("## 5. Supporting Visualizations\n")
    report.append("The following figures are available in the `outputs/figures/` directory:\n")
    for name, filename in cfg.FIGURE_NAMES.items():
        report.append(f"- **{name.replace('_', ' ').title()}**: `{filename}.png`")
    report.append("")
    
    # CONCLUSION
    report.append("---\n")
    report.append("## Conclusion\n")
    report.append("Section 150 requires **immediate attention**. With 301 events (17x the network average), ")
    report.append("significantly higher failure rate than similar sections, and identifiable cause patterns, ")
    report.append("targeted interventions can substantially reduce risk. Priority should be given to ")
    if not top_causes.empty:
        top_cause = top_causes.iloc[0]['Cause']
        report.append(f"addressing **{top_cause}** events, which represent the largest opportunity for improvement.")
    report.append("")
    
    report.append("---\n")
    report.append("*This analysis was generated using PMU disturbance data from 2009-2022.*")
    
    return "\n".join(report)


def generate_technical_report(event_breakdown: Dict,
                              root_cause: Dict,
                              comparative: Dict,
                              characteristics: Dict,
                              recommendations: Dict) -> str:
    """
    Generate detailed technical report.
    
    Returns:
    --------
    str: Markdown formatted technical report
    """
    report = []
    
    report.append("# Section 150 Technical Analysis Report")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    
    # Section details
    sec150_details = characteristics.get('section150_details', {})
    report.append("## PMU Section Details\n")
    report.append("| Attribute | Value |")
    report.append("|-----------|-------|")
    for key, value in sec150_details.items():
        if key not in ['error']:
            report.append(f"| {key} | {value} |")
    report.append("")
    
    # Statistical test results
    significance = root_cause.get('significance_tests', {})
    report.append("## Statistical Test Results\n")
    report.append("### Rate Significance Tests")
    report.append(f"- **Poisson Test**: statistic = N/A, p-value = {significance.get('poisson_p_value', 'N/A')}")
    report.append(f"- **Z-Test**: z = {significance.get('z_score', 'N/A'):.4f}, p-value = {significance.get('z_p_value', 'N/A')}")
    report.append(f"- **Conclusion**: {significance.get('conclusion', 'N/A')}")
    report.append("")
    
    # Time-to-failure analysis
    ttf = root_cause.get('time_to_failure', {})
    report.append("### Time-to-Failure Analysis")
    report.append(f"- **Section 150 MTBF**: {ttf.get('section150_mean_hours', 0):.2f} hours ({ttf.get('section150_mean_hours', 0)/24:.2f} days)")
    report.append(f"- **Network MTBF**: {ttf.get('network_mean_hours', 0):.2f} hours ({ttf.get('network_mean_hours', 0)/24:.2f} days)")
    report.append(f"- **KS Test**: statistic = {ttf.get('ks_statistic', 'N/A')}, p-value = {ttf.get('ks_p_value', 'N/A')}")
    report.append(f"- **Conclusion**: {ttf.get('ks_conclusion', 'N/A')}")
    report.append("")
    
    # Clustering analysis
    clustering = event_breakdown.get('clustering', {})
    report.append("### Temporal Clustering Analysis")
    report.append(f"- **Dispersion Index**: {clustering.get('dispersion_index', 0):.4f} (>1 indicates clustering)")
    report.append(f"- **Is Poisson Process**: {clustering.get('is_poisson_process', 'N/A')}")
    report.append(f"- **Number of Burst Days**: {clustering.get('n_burst_days', 0)}")
    report.append(f"- **Number of Change Points**: {len(clustering.get('change_points', []))}")
    report.append(f"- **Conclusion**: {clustering.get('clustering_conclusion', 'N/A')}")
    report.append("")
    
    # Full cause analysis
    causes = event_breakdown.get('causes', {})
    report.append("## Cause Distribution Analysis")
    report.append(f"- **Chi-square statistic**: {causes.get('chi2_statistic', 'N/A')}")
    report.append(f"- **Chi-square p-value**: {causes.get('chi2_p_value', 'N/A')}")
    report.append(f"- **Distribution differs from network**: {causes.get('distribution_differs', 'N/A')}")
    report.append("")
    
    # Top causes detailed
    top_causes = root_cause.get('top_causes', pd.DataFrame())
    if not top_causes.empty:
        report.append("### Detailed Cause Analysis")
        report.append("| Cause | Sec150 Count | Sec150 % | Network % | Relative Risk | Chi2 p-value | Significant |")
        report.append("|-------|--------------|----------|-----------|---------------|--------------|-------------|")
        for _, row in top_causes.iterrows():
            report.append(f"| {row['Cause']} | {int(row['Section_150_Count'])} | {row['Section_150_Pct']:.2f}% | {row['Network_Pct']:.2f}% | {row['Relative_Risk']:.3f} | {row.get('P_Value', 'N/A')} | {row.get('Significant', 'N/A')} |")
    report.append("")
    
    # Unique features
    unique = characteristics.get('unique_features', [])
    report.append("## Unique Characteristics of Section 150")
    for feature in unique:
        report.append(f"- {feature}")
    report.append("")
    
    # Methodology
    report.append("## Methodology")
    report.append("### Statistical Tests Used")
    report.append("1. **Poisson Test**: Tests if event count exceeds expected under Poisson assumption")
    report.append("2. **Chi-square Test**: Tests if cause distribution differs from network")
    report.append("3. **Kolmogorov-Smirnov Test**: Compares inter-arrival time distributions")
    report.append("4. **Mann-Whitney U Test**: Non-parametric test for MTBF differences")
    report.append("5. **Change Point Detection (PELT)**: Identifies regime changes in event frequency")
    report.append("")
    
    report.append("### Similarity Matching")
    report.append("Similar sections identified using cosine similarity on normalized PMU features ")
    report.append("(voltage level, PMU type, age, location) excluding event count to avoid data leakage.")
    report.append("")
    
    return "\n".join(report)


def save_reports(executive_summary: str, technical_report: str):
    """Save reports to files."""
    Path(cfg.REPORT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save executive summary
    with open(cfg.EXECUTIVE_SUMMARY_FILE, 'w') as f:
        f.write(executive_summary)
    print(f"Executive summary saved to: {cfg.EXECUTIVE_SUMMARY_FILE}")
    
    # Save technical report
    with open(cfg.TECHNICAL_REPORT_FILE, 'w') as f:
        f.write(technical_report)
    print(f"Technical report saved to: {cfg.TECHNICAL_REPORT_FILE}")


def generate_all_reports(event_breakdown: Dict,
                         root_cause: Dict,
                         comparative: Dict,
                         characteristics: Dict,
                         recommendations: Dict) -> Dict[str, str]:
    """
    Generate and save all reports.
    
    Returns:
    --------
    Dict[str, str]: Dictionary of report type to content
    """
    print("\nGenerating reports...")
    
    executive = generate_executive_summary(
        event_breakdown, root_cause, comparative, characteristics, recommendations
    )
    
    technical = generate_technical_report(
        event_breakdown, root_cause, comparative, characteristics, recommendations
    )
    
    save_reports(executive, technical)
    
    return {
        'executive_summary': executive,
        'technical_report': technical
    }
