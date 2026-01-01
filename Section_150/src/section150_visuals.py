"""
Visualizations for Section 150 analysis.
Creates 5-7 publication-quality figures focused on Section 150.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config_section150 as cfg

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


def ensure_output_dirs():
    """Create output directories if they don't exist."""
    Path(cfg.FIGURE_DIR, 'static').mkdir(parents=True, exist_ok=True)
    Path(cfg.FIGURE_DIR, 'interactive').mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename: str, static: bool = True, interactive: bool = True):
    """Save figure in both static and interactive formats."""
    ensure_output_dirs()
    
    is_plotly = isinstance(fig, (go.Figure,))
    
    if is_plotly:
        if interactive:
            html_path = Path(cfg.FIGURE_DIR, 'interactive', f'{filename}.html')
            fig.write_html(str(html_path))
        if static:
            try:
                png_path = Path(cfg.FIGURE_DIR, 'static', f'{filename}.png')
                fig.write_image(str(png_path), width=1400, height=900, scale=2)
            except Exception as e:
                print(f"Could not save static image (install kaleido): {e}")
    else:
        if static:
            png_path = Path(cfg.FIGURE_DIR, 'static', f'{filename}.png')
            pdf_path = Path(cfg.FIGURE_DIR, 'static', f'{filename}.pdf')
            fig.savefig(png_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', facecolor='white')
            fig.savefig(pdf_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def plot_event_timeline(event_breakdown: Dict) -> go.Figure:
    """
    Figure 1: Event timeline with anomalies and change points.
    """
    clustering = event_breakdown['clustering']
    daily_counts = clustering['daily_counts']
    change_points = clustering.get('change_point_dates', [])
    
    fig = go.Figure()
    
    # Main timeline
    fig.add_trace(go.Scatter(
        x=daily_counts.index,
        y=daily_counts.values,
        mode='lines',
        name='Daily Events',
        line=dict(color='#2E86AB', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)'
    ))
    
    # Rolling mean
    rolling_mean = daily_counts.rolling(window=30, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        mode='lines',
        name='30-Day Rolling Mean',
        line=dict(color='#E94F37', width=2.5)
    ))
    
    # Change points
    for cp_date in change_points:
        if cp_date in daily_counts.index:
            fig.add_vline(
                x=cp_date,
                line=dict(color='#F26419', width=2, dash='dash'),
                annotation_text='Change Point',
                annotation_position='top'
            )
    
    # Burst days
    burst_days = clustering.get('burst_days', pd.Series())
    if not burst_days.empty:
        fig.add_trace(go.Scatter(
            x=burst_days.index,
            y=burst_days.values,
            mode='markers',
            name='Burst Days',
            marker=dict(color='#F26419', size=10, symbol='diamond')
        ))
    
    fig.update_layout(
        title='Section 150: Disturbance Event Timeline (2009-2022)',
        xaxis_title='Date',
        yaxis_title='Daily Event Count',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )
    
    return fig


def plot_cause_distribution(event_breakdown: Dict) -> go.Figure:
    """
    Figure 2: Cause distribution - Section 150 vs Network.
    """
    causes = event_breakdown['causes']
    comparison = causes['comparison_df'].head(8)  # Top 8 causes
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Section 150 Causes', 'Section 150 vs Network Average'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    # Pie chart for Section 150
    sec150_causes = causes['section150_causes'].head(8)
    fig.add_trace(
        go.Pie(
            labels=sec150_causes.index,
            values=sec150_causes.values,
            hole=0.4,
            textposition='inside',
            textinfo='percent+label',
            marker=dict(colors=px.colors.qualitative.Set2)
        ),
        row=1, col=1
    )
    
    # Bar comparison
    causes_sorted = comparison.sort_values('Section_150_Pct', ascending=True)
    
    fig.add_trace(
        go.Bar(
            y=causes_sorted.index,
            x=causes_sorted['Section_150_Pct'],
            name='Section 150',
            orientation='h',
            marker_color='#2E86AB',
            text=[f"{v:.1f}%" for v in causes_sorted['Section_150_Pct']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            y=causes_sorted.index,
            x=causes_sorted['Network_Pct'],
            name='Network Average',
            orientation='h',
            marker_color='#A2D729',
            opacity=0.7,
            text=[f"{v:.1f}%" for v in causes_sorted['Network_Pct']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Section 150: Cause Distribution Analysis',
        height=500,
        template='plotly_white',
        barmode='group',
        legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99)
    )
    
    fig.update_xaxes(title_text='Percentage (%)', row=1, col=2)
    
    return fig


def plot_interarrival_analysis(root_cause: Dict) -> go.Figure:
    """
    Figure 3: Inter-arrival time distribution with Poisson fit.
    """
    ttf = root_cause['time_to_failure']
    iat = ttf['section150_iat']
    
    # Convert to days for better readability
    iat_days = iat / 24
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Time Between Events Distribution', 'Section 150 vs Network MTBF')
    )
    
    # Histogram with exponential fit
    fig.add_trace(
        go.Histogram(
            x=iat_days,
            nbinsx=50,
            name='Section 150',
            marker_color='#2E86AB',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add exponential fit line
    x_range = np.linspace(0, iat_days.max(), 100)
    lambda_param = 1 / iat_days.mean()
    exp_fit = len(iat_days) * (iat_days.max() / 50) * lambda_param * np.exp(-lambda_param * x_range)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=exp_fit,
            mode='lines',
            name='Exponential Fit',
            line=dict(color='#E94F37', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # MTBF comparison
    categories = ['Section 150', 'Network Average']
    mtbf_values = [
        ttf['section150_mean_hours'] / 24,  # Convert to days
        ttf['network_mean_hours'] / 24
    ]
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=mtbf_values,
            marker_color=['#E94F37', '#2E86AB'],
            text=[f"{v:.1f} days" for v in mtbf_values],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Section 150: Time-to-Failure Analysis',
        height=450,
        template='plotly_white',
        showlegend=True
    )
    
    fig.update_xaxes(title_text='Days Between Events', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='', row=1, col=2)
    fig.update_yaxes(title_text='Mean Time Between Failures (Days)', row=1, col=2)
    
    return fig


def plot_cyclical_patterns(event_breakdown: Dict) -> go.Figure:
    """
    Figure 4: Cyclical patterns - hourly, daily, monthly heatmap.
    """
    patterns = event_breakdown['cyclical_patterns']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hourly Pattern', 'Day of Week', 'Monthly Pattern', 'Yearly Trend'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Hourly pattern
    hourly = patterns['hourly']
    fig.add_trace(
        go.Bar(
            x=hourly.index,
            y=hourly.values,
            marker_color='#2E86AB',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Daily pattern
    daily = patterns['daily']
    colors = ['#A2D729' if d in ['Saturday', 'Sunday'] else '#2E86AB' for d in daily.index]
    fig.add_trace(
        go.Bar(
            x=daily.index,
            y=daily.values,
            marker_color=colors,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Monthly pattern
    monthly = patterns['monthly']
    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly.values,
            marker_color='#2E86AB',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Yearly trend (if available in breakdown)
    root_cause = event_breakdown.get('root_cause', {})
    seasonal = root_cause.get('seasonal_patterns', {})
    yearly = seasonal.get('yearly_trend', pd.Series())
    
    if yearly.empty:
        # Calculate from timeline
        timeline = event_breakdown.get('timeline', pd.DataFrame())
        if not timeline.empty and 'Year' in timeline.columns:
            yearly = timeline.groupby('Year').size()
    
    if not yearly.empty:
        fig.add_trace(
            go.Scatter(
                x=yearly.index.astype(str),
                y=yearly.values,
                mode='lines+markers',
                marker=dict(size=10, color='#E94F37'),
                line=dict(width=2, color='#E94F37'),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title='Section 150: Temporal Patterns',
        height=600,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Hour', row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(title_text='Year', row=2, col=2)
    
    return fig


def plot_similar_sections_comparison(comparative: Dict) -> go.Figure:
    """
    Figure 5: Section 150 vs similar sections comparison.
    """
    similar = comparative.get('similar_sections', pd.DataFrame())
    sec150_profile = comparative.get('section150_profile', {})
    
    if similar.empty:
        fig = go.Figure()
        fig.add_annotation(text="No similar sections data available", showarrow=False)
        return fig
    
    # Prepare data
    sections = ['Section 150'] + [f"Section {int(s)}" for s in similar['section_id'].values[:9]]
    events = [sec150_profile.get('event_count', 301)] + similar['event_count'].values[:9].tolist()
    
    colors = ['#E94F37'] + ['#2E86AB'] * (len(sections) - 1)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=sections,
            y=events,
            marker_color=colors,
            text=[f"{int(e)}" for e in events],
            textposition='auto'
        )
    )
    
    # Add network average line
    network_avg = sum(events[1:]) / (len(events) - 1) if len(events) > 1 else 0
    fig.add_hline(
        y=network_avg,
        line=dict(color='#A2D729', width=2, dash='dash'),
        annotation_text=f'Similar Sections Avg: {network_avg:.0f}',
        annotation_position='right'
    )
    
    fig.update_layout(
        title='Section 150 vs Similar Sections (Same Voltage & Type)',
        xaxis_title='Section',
        yaxis_title='Total Disturbance Events',
        height=450,
        template='plotly_white'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def plot_pmu_characteristics(characteristics: Dict) -> go.Figure:
    """
    Figure 6: PMU characteristics comparison (radar/spider chart).
    """
    sec150 = characteristics.get('section150_details', {})
    similar = characteristics.get('similar_sections', pd.DataFrame())
    
    # Create comparison metrics
    categories = ['Event Count', 'Age (Years)', 'Relative Risk']
    
    sec150_values = [
        sec150.get('Event_Count', 301),
        sec150.get('Age_Years', 0),
        sec150.get('Event_Count', 301) / similar['Event_Count'].mean() if not similar.empty else 1
    ]
    
    # Normalize values to 0-100 scale for radar chart
    max_vals = [
        max(sec150_values[0], similar['Event_Count'].max() if not similar.empty else 1),
        max(sec150_values[1], 20),  # Assume max age 20 years
        max(sec150_values[2], 5)
    ]
    
    sec150_norm = [v / m * 100 for v, m in zip(sec150_values, max_vals)]
    
    if not similar.empty:
        similar_avg_values = [
            similar['Event_Count'].mean(),
            10,  # Placeholder for average age
            1.0
        ]
        similar_norm = [v / m * 100 for v, m in zip(similar_avg_values, max_vals)]
    else:
        similar_norm = [50, 50, 50]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=sec150_norm + [sec150_norm[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Section 150',
        line_color='#E94F37',
        fillcolor='rgba(233, 79, 55, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=similar_norm + [similar_norm[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Similar Sections Avg',
        line_color='#2E86AB',
        fillcolor='rgba(46, 134, 171, 0.3)'
    ))
    
    fig.update_layout(
        title='Section 150: Characteristics Comparison',
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=450,
        template='plotly_white'
    )
    
    return fig


def plot_cumulative_events(event_breakdown: Dict) -> go.Figure:
    """
    Figure 7: Cumulative events with change point annotations.
    """
    clustering = event_breakdown['clustering']
    daily_counts = clustering['daily_counts']
    change_points = clustering.get('change_point_dates', [])
    
    # Calculate cumulative
    cumulative = daily_counts.cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative.values,
        mode='lines',
        name='Cumulative Events',
        line=dict(color='#2E86AB', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.1)'
    ))
    
    # Add change points
    for i, cp_date in enumerate(change_points):
        if cp_date in cumulative.index:
            cp_value = cumulative.loc[cp_date]
            fig.add_trace(go.Scatter(
                x=[cp_date],
                y=[cp_value],
                mode='markers+text',
                name=f'Change Point {i+1}',
                marker=dict(size=12, color='#E94F37', symbol='star'),
                text=[f'CP{i+1}'],
                textposition='top center'
            ))
    
    # Add expected line (linear growth at network average rate)
    start_date = cumulative.index[0]
    end_date = cumulative.index[-1]
    days_total = (end_date - start_date).days
    expected_rate = 17.6  # Network average events per section
    expected_daily = expected_rate / 365
    expected_line = pd.Series(
        [expected_daily * i for i in range(len(cumulative))],
        index=cumulative.index
    )
    
    fig.add_trace(go.Scatter(
        x=expected_line.index,
        y=expected_line.values,
        mode='lines',
        name='Expected (Network Avg Rate)',
        line=dict(color='#A2D729', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Section 150: Cumulative Event Growth',
        xaxis_title='Date',
        yaxis_title='Cumulative Events',
        height=450,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def plot_cromwell_tap_analysis(operational_context: Dict) -> go.Figure:
    """
    Figure 8: Cromwell Tap 31 operational state analysis.

    Layout:
    - Left: Pie chart of state distribution
    - Right: Bar chart of event rate by state
    """
    from plotly.subplots import make_subplots

    cromwell = operational_context.get('cromwell', {})
    state_dist = cromwell.get('state_distribution', {})

    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cromwell Tap 31 State Distribution', 'Event Count by State'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Pie chart
    states = list(state_dist.keys())
    counts = list(state_dist.values())

    fig.add_trace(go.Pie(
        labels=states,
        values=counts,
        marker=dict(colors=['#E94F37', '#2E86AB', '#F26419', '#8B8C89']),
        textinfo='label+percent',
        hovertemplate='%{label}<br>%{value} events<br>%{percent}<extra></extra>'
    ), row=1, col=1)

    # Bar chart
    fig.add_trace(go.Bar(
        x=states,
        y=counts,
        marker_color=['#E94F37', '#2E86AB', '#F26419', '#8B8C89'],
        text=counts,
        textposition='outside',
        hovertemplate='%{x}<br>%{y} events<extra></extra>'
    ), row=1, col=2)

    fig.update_layout(
        title=f'Cromwell Tap 31 Operational Context ({cromwell.get("mentions", 0)} total mentions)',
        height=450,
        template='plotly_white',
        showlegend=False
    )

    fig.update_yaxes(title_text='Event Count', row=1, col=2)

    return fig


def plot_unknown_reclassification(unknown_analysis: Dict) -> go.Figure:
    """
    Figure 9: Unknown events reclassification analysis.

    Sankey diagram showing Unknown → Suggested Category flows.
    """
    reclassification = unknown_analysis.get('reclassification', {})
    suggested_dist = reclassification.get('suggested_distribution', {})

    # Prepare Sankey data
    source_labels = ['Unknown']
    target_labels = list(suggested_dist.keys())
    all_labels = source_labels + target_labels

    # Create indices
    source_indices = [0] * len(target_labels)  # All from "Unknown"
    target_indices = list(range(1, len(all_labels)))  # To each suggested category
    values = list(suggested_dist.values())

    # Color coding
    colors = {
        'Weather, excluding lightning': '#2E86AB',
        'Maintenance/Testing': '#F26419',
        'Equipment/Other': '#8B8C89',
        'Insufficient clues': '#CCCCCC'
    }

    link_colors = [colors.get(label, '#CCCCCC') for label in target_labels]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=all_labels,
            color=['#E94F37'] + [colors.get(label, '#CCCCCC') for label in target_labels]
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=[f"rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.4)" for c in link_colors]
        )
    ))

    fig.update_layout(
        title=f'Unknown Events Reclassification Flow ({reclassification.get("total_unknown", 0)} events)',
        height=500,
        template='plotly_white'
    )

    return fig


def plot_hourly_comparison(hourly_analysis: Dict) -> go.Figure:
    """
    Figure 10: Section 150 vs Network hourly distribution.

    Dual-axis: Section 150 hourly bars + Network hourly line.
    """
    hourly_comp = hourly_analysis.get('hourly_comparison', {})

    sec150_hourly = hourly_comp.get('section150_hourly_pct', pd.Series())
    network_hourly = hourly_comp.get('network_hourly_pct', pd.Series())

    fig = go.Figure()

    # Section 150 bars
    fig.add_trace(go.Bar(
        x=sec150_hourly.index,
        y=sec150_hourly.values,
        name='Section 150',
        marker_color='#E94F37',
        opacity=0.7,
        yaxis='y',
        hovertemplate='Hour %{x}<br>%{y:.1f}%<extra>Section 150</extra>'
    ))

    # Network line
    fig.add_trace(go.Scatter(
        x=network_hourly.index,
        y=network_hourly.values,
        name='Network Average',
        line=dict(color='#2E86AB', width=3),
        mode='lines+markers',
        yaxis='y',
        hovertemplate='Hour %{x}<br>%{y:.1f}%<extra>Network</extra>'
    ))

    # Highlight hour 19
    hour_19_pct = hourly_comp.get('hour_19_pct_150', 0)
    fig.add_vline(
        x=19,
        line=dict(color='#F26419', width=2, dash='dash'),
        annotation_text=f'Peak: 7 PM ({hour_19_pct:.1f}%)',
        annotation_position='top'
    )

    fig.add_vrect(
        x0=18.5, x1=19.5,
        fillcolor='#F26419',
        opacity=0.1,
        line_width=0
    )

    fig.update_layout(
        title='Section 150 vs Network: Hourly Event Distribution',
        xaxis_title='Hour of Day',
        yaxis_title='Percentage of Events (%)',
        height=450,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )

    fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)

    return fig


def plot_7pm_deep_dive(hourly_analysis: Dict) -> go.Figure:
    """
    Figure 11: 7 PM peak multi-factor analysis.

    2x2 subplots showing different factors at hour 19.
    """
    from plotly.subplots import make_subplots

    cause_breakdown = hourly_analysis.get('cause_breakdown_7pm', pd.DataFrame())
    ops_analysis = hourly_analysis.get('operations_by_hour', {})
    cromwell_analysis = hourly_analysis.get('cromwell_by_hour', {})
    unknown_timing = hourly_analysis.get('unknown_timing', {})

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cause Breakdown: 7 PM vs Other Hours',
            'Operations Field Distribution',
            'Cromwell Tap State',
            'Unknown Events Hourly Distribution'
        ),
        specs=[[{'type': 'bar'}, {'type': 'pie'}],
               [{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Subplot 1: Cause breakdown
    if not cause_breakdown.empty:
        top_causes = cause_breakdown.head(5)
        fig.add_trace(go.Bar(
            x=top_causes['Cause_Category'],
            y=top_causes['Count_At_19'],
            name='At 7 PM',
            marker_color='#E94F37',
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=top_causes['Cause_Category'],
            y=top_causes['Count_Other_Hours'],
            name='Other Hours',
            marker_color='#2E86AB',
            showlegend=True
        ), row=1, col=1)

    # Subplot 2: Operations field
    hour_19_ops = ops_analysis.get('hour_19_ops_dist', {})
    if hour_19_ops:
        fig.add_trace(go.Pie(
            labels=[f"Ops {k}" for k in hour_19_ops.keys()],
            values=list(hour_19_ops.values()),
            marker=dict(colors=['#E94F37', '#2E86AB', '#F26419']),
            textinfo='label+percent',
            showlegend=False
        ), row=1, col=2)

    # Subplot 3: Cromwell Tap state
    hour_19_cromwell = cromwell_analysis.get('hour_19_state_distribution', {})
    if hour_19_cromwell:
        fig.add_trace(go.Pie(
            labels=list(hour_19_cromwell.keys()),
            values=list(hour_19_cromwell.values()),
            marker=dict(colors=['#E94F37', '#2E86AB', '#F26419', '#8B8C89']),
            textinfo='label+percent',
            showlegend=False
        ), row=2, col=1)

    # Subplot 4: Unknown timing
    unknown_hourly = unknown_timing.get('unknown_hourly_distribution', pd.Series())
    if not unknown_hourly.empty:
        fig.add_trace(go.Bar(
            x=unknown_hourly.index,
            y=unknown_hourly.values,
            marker_color=['#E94F37' if h == 19 else '#2E86AB' for h in unknown_hourly.index],
            showlegend=False,
            hovertemplate='Hour %{x}<br>%{y} events<extra></extra>'
        ), row=2, col=2)

    fig.update_layout(
        title='7 PM Peak: Multi-Factor Deep Dive',
        height=700,
        template='plotly_white',
        showlegend=True
    )

    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text='Event Count', row=1, col=1)
    fig.update_yaxes(title_text='Event Count', row=2, col=2)
    fig.update_xaxes(title_text='Hour', row=2, col=2)

    return fig


def plot_day_hour_heatmap(hourly_analysis: Dict) -> go.Figure:
    """
    Figure 12: Day of week x Hour of day heatmap.

    7 rows (Mon-Sun) x 24 columns (0-23 hours).
    """
    day_hour_interaction = hourly_analysis.get('day_hour_interaction', {})
    day_hour_matrix = day_hour_interaction.get('day_hour_matrix', pd.DataFrame())

    if day_hour_matrix.empty:
        # Create empty matrix
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_hour_matrix = pd.DataFrame(0, index=day_names, columns=range(24))

    fig = go.Figure(go.Heatmap(
        z=day_hour_matrix.values,
        x=day_hour_matrix.columns,
        y=day_hour_matrix.index,
        colorscale='RdYlBu_r',
        text=day_hour_matrix.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y}<br>Hour %{x}<br>%{z} events<extra></extra>',
        colorbar=dict(title='Event<br>Count')
    ))

    # Find peak cell
    peak_day = day_hour_interaction.get('peak_day_at_19')
    if peak_day and 19 in day_hour_matrix.columns:
        fig.add_shape(
            type='rect',
            x0=18.5, x1=19.5,
            y0=list(day_hour_matrix.index).index(peak_day) - 0.5,
            y1=list(day_hour_matrix.index).index(peak_day) + 0.5,
            line=dict(color='#E94F37', width=3)
        )

    fig.update_layout(
        title='Day of Week × Hour of Day: Event Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=500,
        template='plotly_white'
    )

    fig.update_xaxes(side='top', tickmode='linear', tick0=0, dtick=2)

    return fig


def generate_extended_visualizations(operational_context: Dict,
                                      unknown_analysis: Dict,
                                      hourly_analysis: Dict) -> Dict[str, go.Figure]:
    """
    Generate 5 extended visualizations for Section 150 analysis.

    Figures 8-12 for operational context, unknown analysis, and hourly patterns.

    Returns:
    --------
    Dict[str, go.Figure]: Dictionary of figure name to figure object
    """
    ensure_output_dirs()

    figures = {}

    print("Generating Figure 8: Cromwell Tap Analysis...")
    figures['cromwell_analysis'] = plot_cromwell_tap_analysis(operational_context)
    save_figure(figures['cromwell_analysis'], cfg.FIGURE_NAMES['cromwell_analysis'])

    print("Generating Figure 9: Unknown Reclassification...")
    figures['unknown_reclassification'] = plot_unknown_reclassification(unknown_analysis)
    save_figure(figures['unknown_reclassification'], cfg.FIGURE_NAMES['unknown_reclassification'])

    print("Generating Figure 10: Hourly Comparison...")
    figures['hourly_comparison'] = plot_hourly_comparison(hourly_analysis)
    save_figure(figures['hourly_comparison'], cfg.FIGURE_NAMES['hourly_comparison'])

    print("Generating Figure 11: 7 PM Deep Dive...")
    figures['7pm_deep_dive'] = plot_7pm_deep_dive(hourly_analysis)
    save_figure(figures['7pm_deep_dive'], cfg.FIGURE_NAMES['7pm_deep_dive'])

    print("Generating Figure 12: Day-Hour Heatmap...")
    figures['day_hour_heatmap'] = plot_day_hour_heatmap(hourly_analysis)
    save_figure(figures['day_hour_heatmap'], cfg.FIGURE_NAMES['day_hour_heatmap'])

    print(f"\nExtended figures saved to {cfg.FIGURE_DIR}")

    return figures


def generate_all_visualizations(event_breakdown: Dict,
                                root_cause: Dict,
                                comparative: Dict,
                                characteristics: Dict) -> Dict[str, go.Figure]:
    """
    Generate all 7 visualizations for Section 150 analysis.
    
    Returns:
    --------
    Dict[str, go.Figure]: Dictionary of figure name to figure object
    """
    ensure_output_dirs()
    
    # Add root_cause to event_breakdown for cyclical patterns figure
    event_breakdown['root_cause'] = root_cause
    
    figures = {}
    
    print("Generating Figure 1: Event Timeline...")
    figures['timeline'] = plot_event_timeline(event_breakdown)
    save_figure(figures['timeline'], cfg.FIGURE_NAMES['timeline'])
    
    print("Generating Figure 2: Cause Distribution...")
    figures['cause_distribution'] = plot_cause_distribution(event_breakdown)
    save_figure(figures['cause_distribution'], cfg.FIGURE_NAMES['cause_distribution'])
    
    print("Generating Figure 3: Inter-arrival Analysis...")
    figures['interarrival'] = plot_interarrival_analysis(root_cause)
    save_figure(figures['interarrival'], cfg.FIGURE_NAMES['interarrival'])
    
    print("Generating Figure 4: Cyclical Patterns...")
    figures['cyclical_patterns'] = plot_cyclical_patterns(event_breakdown)
    save_figure(figures['cyclical_patterns'], cfg.FIGURE_NAMES['cyclical_patterns'])
    
    print("Generating Figure 5: Similar Sections Comparison...")
    figures['similar_sections'] = plot_similar_sections_comparison(comparative)
    save_figure(figures['similar_sections'], cfg.FIGURE_NAMES['similar_sections'])
    
    print("Generating Figure 6: PMU Characteristics...")
    figures['pmu_characteristics'] = plot_pmu_characteristics(characteristics)
    save_figure(figures['pmu_characteristics'], cfg.FIGURE_NAMES['pmu_characteristics'])
    
    print("Generating Figure 7: Cumulative Events...")
    figures['cumulative_events'] = plot_cumulative_events(event_breakdown)
    save_figure(figures['cumulative_events'], cfg.FIGURE_NAMES['cumulative_events'])
    
    print(f"\nAll figures saved to {cfg.FIGURE_DIR}")
    
    return figures
