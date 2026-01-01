"""
Recommendations generator for Section 150.
Generates actionable maintenance interventions based on analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_section150 as cfg


def prioritize_causes(top_causes: pd.DataFrame) -> List[Dict]:
    """
    Prioritize causes for intervention based on frequency and relative risk.
    
    Returns:
    --------
    List[Dict]: Prioritized causes with intervention recommendations
    """
    priorities = []
    
    for _, row in top_causes.iterrows():
        cause = row['Cause']
        count = row['Section_150_Count']
        pct = row['Section_150_Pct']
        rr = row['Relative_Risk']
        significant = row.get('Significant', False)
        
        # Priority score: combination of frequency and relative risk
        priority_score = (pct / 100) * np.log1p(rr) * (1.5 if significant else 1.0)
        
        # Generate intervention based on cause
        intervention = get_cause_intervention(cause)
        
        priorities.append({
            'cause': cause,
            'event_count': count,
            'percentage': pct,
            'relative_risk': rr,
            'priority_score': priority_score,
            'statistically_significant': significant,
            'intervention': intervention
        })
    
    # Sort by priority score
    priorities.sort(key=lambda x: x['priority_score'], reverse=True)
    
    return priorities


def get_cause_intervention(cause: str) -> Dict:
    """
    Get specific intervention recommendations for a cause.
    
    Returns:
    --------
    Dict: Intervention details
    """
    interventions = {
        'Weather': {
            'action': 'Install weather-resistant enclosures and lightning arresters',
            'frequency': 'Before storm season',
            'expected_reduction': '20-30%',
            'cost_estimate': 'Medium'
        },
        'Lightning': {
            'action': 'Install surge protection and improve grounding systems',
            'frequency': 'Annual inspection before lightning season',
            'expected_reduction': '30-40%',
            'cost_estimate': 'Medium-High'
        },
        'Equipment Failure': {
            'action': 'Implement predictive maintenance program with regular inspections',
            'frequency': 'Monthly inspections, quarterly deep maintenance',
            'expected_reduction': '25-35%',
            'cost_estimate': 'High'
        },
        'Wildlife': {
            'action': 'Install wildlife guards and deterrent systems',
            'frequency': 'One-time installation, annual inspection',
            'expected_reduction': '40-60%',
            'cost_estimate': 'Low-Medium'
        },
        'Vegetation': {
            'action': 'Implement aggressive vegetation management program',
            'frequency': 'Quarterly trimming, annual clearing',
            'expected_reduction': '50-70%',
            'cost_estimate': 'Medium'
        },
        'Human Error': {
            'action': 'Enhanced operator training and procedure automation',
            'frequency': 'Quarterly training sessions',
            'expected_reduction': '20-40%',
            'cost_estimate': 'Low'
        },
        'Unknown': {
            'action': 'Deploy enhanced monitoring and event logging systems',
            'frequency': 'Continuous monitoring with monthly reviews',
            'expected_reduction': '15-25% (through better diagnosis)',
            'cost_estimate': 'Medium'
        }
    }
    
    # Default intervention for causes not in the list
    default = {
        'action': 'Conduct detailed root cause investigation',
        'frequency': 'After each incident',
        'expected_reduction': 'TBD after investigation',
        'cost_estimate': 'Low'
    }
    
    # Try to match cause to known patterns
    for key in interventions:
        if key.lower() in cause.lower():
            return interventions[key]
    
    return default


def generate_monitoring_recommendations(seasonal_patterns: Dict,
                                         time_to_failure: Dict) -> Dict:
    """
    Generate monitoring frequency recommendations based on patterns.
    
    Returns:
    --------
    Dict: Monitoring recommendations
    """
    recommendations = {
        'base_frequency': 'Daily automated monitoring',
        'enhanced_periods': [],
        'alert_thresholds': {}
    }
    
    # Peak hour monitoring
    peak_hour = seasonal_patterns.get('peak_hour')
    if peak_hour is not None:
        recommendations['enhanced_periods'].append(
            f"Increase monitoring during peak hour ({peak_hour}:00)"
        )
    
    # Peak month monitoring
    peak_month = seasonal_patterns.get('peak_month')
    if peak_month:
        recommendations['enhanced_periods'].append(
            f"Deploy additional resources during {peak_month}"
        )
    
    # Weekend effect
    weekend_effect = seasonal_patterns.get('weekend_effect', '')
    if 'Higher on weekends' in weekend_effect:
        recommendations['enhanced_periods'].append(
            "Maintain staffing levels on weekends"
        )
    
    # Based on inter-arrival times
    mean_iat = time_to_failure.get('section150_mean_hours')
    if mean_iat:
        # Set alert threshold at 25% of mean inter-arrival time
        recommendations['alert_thresholds']['rapid_succession'] = mean_iat * 0.25
        recommendations['alert_thresholds']['unusual_quiet'] = mean_iat * 2
        
        if mean_iat < 24 * 7:  # Less than a week on average
            recommendations['base_frequency'] = 'Multiple daily checks recommended'
        elif mean_iat < 24 * 30:  # Less than a month
            recommendations['base_frequency'] = 'Daily monitoring recommended'
        else:
            recommendations['base_frequency'] = 'Weekly monitoring sufficient'
    
    return recommendations


def estimate_risk_reduction(prioritized_causes: List[Dict],
                            section150_events: int) -> Dict:
    """
    Estimate potential risk reduction from implementing interventions.
    
    Returns:
    --------
    Dict: Risk reduction estimates
    """
    total_reducible_events = 0
    reduction_scenarios = []
    
    for cause_info in prioritized_causes[:5]:  # Top 5 causes
        cause = cause_info['cause']
        count = cause_info['event_count']
        intervention = cause_info['intervention']
        
        # Parse expected reduction
        reduction_str = intervention.get('expected_reduction', '0%')
        try:
            # Extract middle of range if given as range
            if '-' in reduction_str:
                parts = reduction_str.replace('%', '').split('-')
                reduction_pct = (float(parts[0]) + float(parts[1])) / 2 / 100
            else:
                reduction_pct = float(reduction_str.replace('%', '')) / 100
        except:
            reduction_pct = 0.2  # Default 20%
        
        events_prevented = count * reduction_pct
        total_reducible_events += events_prevented
        
        reduction_scenarios.append({
            'cause': cause,
            'current_events': count,
            'estimated_reduction_pct': reduction_pct * 100,
            'events_preventable': events_prevented,
            'intervention': intervention['action']
        })
    
    return {
        'total_events': section150_events,
        'total_preventable': total_reducible_events,
        'overall_reduction_pct': (total_reducible_events / section150_events * 100) if section150_events > 0 else 0,
        'scenarios': reduction_scenarios
    }


def generate_cost_benefit_analysis(risk_reduction: Dict,
                                   prioritized_causes: List[Dict]) -> Dict:
    """
    Generate simplified cost-benefit analysis.
    
    Returns:
    --------
    Dict: Cost-benefit summary
    """
    cost_levels = {'Low': 1, 'Low-Medium': 2, 'Medium': 3, 'Medium-High': 4, 'High': 5}
    
    interventions = []
    for cause_info in prioritized_causes[:5]:
        intervention = cause_info['intervention']
        cost = intervention.get('cost_estimate', 'Medium')
        reduction = intervention.get('expected_reduction', '20%')
        
        cost_score = cost_levels.get(cost, 3)
        
        # Calculate benefit score (inverse of cost relative to reduction)
        try:
            if '-' in reduction:
                parts = reduction.replace('%', '').split('-')
                reduction_val = (float(parts[0]) + float(parts[1])) / 2
            else:
                reduction_val = float(reduction.replace('%', '').replace('TBD after investigation', '15'))
        except:
            reduction_val = 20
        
        benefit_score = reduction_val / cost_score
        
        interventions.append({
            'cause': cause_info['cause'],
            'intervention': intervention['action'],
            'cost_level': cost,
            'expected_reduction': reduction,
            'benefit_score': benefit_score,
            'recommendation': 'High Priority' if benefit_score > 10 else ('Medium Priority' if benefit_score > 5 else 'Low Priority')
        })
    
    # Sort by benefit score
    interventions.sort(key=lambda x: x['benefit_score'], reverse=True)
    
    return {
        'interventions': interventions,
        'highest_roi': interventions[0] if interventions else None,
        'total_investment_level': 'Medium-High' if len(interventions) > 3 else 'Medium'
    }


def get_recommendations_summary(event_breakdown: Dict,
                                root_cause: Dict,
                                comparative: Dict,
                                characteristics: Dict) -> Dict:
    """
    Generate complete recommendations for Section 150.
    
    Returns:
    --------
    Dict: All recommendations
    """
    # Prioritize causes
    top_causes = root_cause.get('top_causes', pd.DataFrame())
    prioritized = prioritize_causes(top_causes) if not top_causes.empty else []
    
    # Monitoring recommendations
    monitoring = generate_monitoring_recommendations(
        root_cause.get('seasonal_patterns', {}),
        root_cause.get('time_to_failure', {})
    )
    
    # Risk reduction estimates
    section150_events = event_breakdown.get('network_comparison', {}).get('section150_events', 301)
    risk_reduction = estimate_risk_reduction(prioritized, section150_events)
    
    # Cost-benefit analysis
    cost_benefit = generate_cost_benefit_analysis(risk_reduction, prioritized)
    
    # Key recommendations (executive summary)
    key_recommendations = []
    
    # Add top intervention
    if prioritized:
        top = prioritized[0]
        key_recommendations.append({
            'priority': 1,
            'recommendation': f"Address {top['cause']} events ({top['percentage']:.1f}% of total)",
            'action': top['intervention']['action'],
            'expected_impact': top['intervention']['expected_reduction']
        })
    
    # Add monitoring recommendation
    key_recommendations.append({
        'priority': 2,
        'recommendation': "Enhance monitoring during peak periods",
        'action': monitoring['base_frequency'],
        'expected_impact': 'Early detection of event clusters'
    })
    
    # Add learning from comparative analysis
    learnings = comparative.get('learnings', [])
    if learnings:
        key_recommendations.append({
            'priority': 3,
            'recommendation': "Apply best practices from similar sections",
            'action': learnings[0] if learnings else "Conduct comparative study",
            'expected_impact': 'Potential 20-40% event reduction'
        })
    
    return {
        'prioritized_causes': prioritized,
        'monitoring_recommendations': monitoring,
        'risk_reduction_estimate': risk_reduction,
        'cost_benefit_analysis': cost_benefit,
        'key_recommendations': key_recommendations,
        'executive_summary': generate_executive_recommendations(key_recommendations, risk_reduction)
    }


def generate_executive_recommendations(key_recommendations: List[Dict],
                                       risk_reduction: Dict) -> str:
    """
    Generate executive-level recommendation summary.
    
    Returns:
    --------
    str: Executive summary text
    """
    summary = []
    summary.append("## Executive Recommendations for Section 150\n")
    
    summary.append(f"**Potential Impact**: Implementing top interventions could reduce events by approximately "
                   f"{risk_reduction.get('overall_reduction_pct', 0):.0f}% "
                   f"({risk_reduction.get('total_preventable', 0):.0f} events preventable)\n")
    
    summary.append("\n### Priority Actions:\n")
    for rec in key_recommendations:
        summary.append(f"{rec['priority']}. **{rec['recommendation']}**")
        summary.append(f"   - Action: {rec['action']}")
        summary.append(f"   - Expected Impact: {rec['expected_impact']}\n")
    
    return "\n".join(summary)
