"""
Visualization Module for PMU Reliability Framework.

Creates publication-quality figures for PMU disturbance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Any

try:
    from .data_loader import get_section_events, calculate_event_statistics, _find_datetime_column, _find_cause_column
    from .temporal_analysis import TemporalAnalyzer
except ImportError:
    from data_loader import get_section_events, calculate_event_statistics, _find_datetime_column, _find_cause_column
    from temporal_analysis import TemporalAnalyzer


plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = sns.color_palette("husl", 10)


def plot_event_timeline(events_df: pd.DataFrame, section_id: int, 
                        save_path: Optional[str] = None,
                        show_trend: bool = True,
                        rolling_window: int = 30) -> plt.Figure:
    """
    Plot event timeline with optional trend line.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events for the section
    section_id : int
        Section ID for title
    save_path : str, optional
        Path to save figure
    show_trend : bool
        Show rolling average trend line
    rolling_window : int
        Rolling window size in days
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    datetime_col = _find_datetime_column(events_df)
    if datetime_col is None:
        raise ValueError("No datetime column found")
    
    events_df = events_df.copy()
    events_df['date'] = events_df[datetime_col].dt.date
    daily_counts = events_df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    full_range = pd.date_range(
        start=daily_counts['date'].min(),
        end=daily_counts['date'].max(),
        freq='D'
    )
    daily_counts = daily_counts.set_index('date').reindex(full_range, fill_value=0).reset_index()
    daily_counts.columns = ['date', 'count']
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    ax.bar(daily_counts['date'], daily_counts['count'], alpha=0.5, color=COLOR_PALETTE[0], label='Daily Events')
    
    if show_trend:
        rolling_avg = daily_counts['count'].rolling(window=rolling_window, center=True).mean()
        ax.plot(daily_counts['date'], rolling_avg, color=COLOR_PALETTE[1], 
                linewidth=2, label=f'{rolling_window}-day Rolling Average')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Event Count', fontsize=12)
    ax.set_title(f'Section {section_id}: Disturbance Event Timeline', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    
    return fig


def plot_cause_distribution(events_df: pd.DataFrame, section_id: int,
                            save_path: Optional[str] = None,
                            top_n: int = 10) -> plt.Figure:
    """
    Plot cause distribution as bar chart.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events for the section
    section_id : int
        Section ID for title
    save_path : str, optional
        Path to save figure
    top_n : int
        Number of top causes to show
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    cause_col = _find_cause_column(events_df)
    if cause_col is None:
        raise ValueError("No cause column found")
    
    cause_counts = events_df[cause_col].value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    bars = ax.barh(cause_counts.index[::-1], cause_counts.values[::-1], color=COLOR_PALETTE)
    
    for bar, count in zip(bars, cause_counts.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count} ({count/len(events_df)*100:.1f}%)', va='center', fontsize=10)
    
    ax.set_xlabel('Event Count', fontsize=12)
    ax.set_ylabel('Cause', fontsize=12)
    ax.set_title(f'Section {section_id}: Disturbance Cause Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    
    return fig


def plot_temporal_heatmap(events_df: pd.DataFrame, section_id: int,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot temporal patterns as heatmaps (hour x day, month x year).
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events for the section
    section_id : int
        Section ID for title
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    analyzer = TemporalAnalyzer(events_df)
    datetime_col = analyzer.datetime_col
    
    events_df = events_df.copy()
    events_df['hour'] = events_df[datetime_col].dt.hour
    events_df['dayofweek'] = events_df[datetime_col].dt.dayofweek
    
    hour_day = events_df.groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)
    hour_day.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    sns.heatmap(hour_day, cmap='YlOrRd', annot=False, ax=ax, cbar_kws={'label': 'Event Count'})
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    ax.set_title(f'Section {section_id}: Temporal Pattern (Hour x Day)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    
    return fig


def plot_risk_distribution(risk_results: pd.DataFrame,
                           save_path: Optional[str] = None,
                           highlight_sections: Optional[List[int]] = None) -> plt.Figure:
    """
    Plot risk score distribution across all sections.
    
    Parameters
    ----------
    risk_results : pd.DataFrame
        Risk scoring results
    save_path : str, optional
        Path to save figure
    highlight_sections : List[int], optional
        Section IDs to highlight
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(risk_results['risk_score'], bins=30, color=COLOR_PALETTE[0], 
                 edgecolor='white', alpha=0.7)
    axes[0].set_xlabel('Risk Score', fontsize=12)
    axes[0].set_ylabel('Number of Sections', fontsize=12)
    axes[0].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    
    if highlight_sections:
        for sid in highlight_sections:
            score = risk_results[risk_results['SectionID'] == sid]['risk_score'].values
            if len(score) > 0:
                axes[0].axvline(x=score[0], color='red', linestyle='--', 
                               label=f'Section {sid}')
        axes[0].legend()
    
    category_counts = risk_results['category'].value_counts()
    colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
    axes[1].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%',
                colors=[colors.get(c, 'gray') for c in category_counts.index])
    axes[1].set_title('Risk Categories', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    
    return fig


def plot_mtbf_histogram(mtbf_values: pd.Series,
                        save_path: Optional[str] = None,
                        highlight_value: Optional[float] = None) -> plt.Figure:
    """
    Plot MTBF distribution histogram.
    
    Parameters
    ----------
    mtbf_values : pd.Series
        MTBF values in days
    save_path : str, optional
        Path to save figure
    highlight_value : float, optional
        MTBF value to highlight
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    mtbf_clean = mtbf_values.replace([np.inf, -np.inf], np.nan).dropna()
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    ax.hist(mtbf_clean, bins=50, color=COLOR_PALETTE[0], edgecolor='white', alpha=0.7)
    
    if highlight_value:
        ax.axvline(x=highlight_value, color='red', linestyle='--', linewidth=2,
                   label=f'Section Value: {highlight_value:.1f} days')
        ax.legend()
    
    ax.set_xlabel('Mean Time Between Failures (days)', fontsize=12)
    ax.set_ylabel('Number of Sections', fontsize=12)
    ax.set_title('MTBF Distribution Across Network', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    
    return fig


def plot_comparative_bar(target_section: int, similar_sections: pd.DataFrame,
                         metric: str = 'Event_Count',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison bar chart between target and similar sections.
    
    Parameters
    ----------
    target_section : int
        Target section ID
    similar_sections : pd.DataFrame
        Similar sections data
    metric : str
        Metric to compare
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    sections = similar_sections['SectionID'].tolist()
    values = similar_sections[metric].tolist()
    
    colors = [COLOR_PALETTE[0]] * len(sections)
    
    bars = ax.bar(range(len(sections)), values, color=colors, edgecolor='white')
    
    ax.set_xticks(range(len(sections)))
    ax.set_xticklabels([f'Sec {s}' for s in sections], rotation=45, ha='right')
    ax.set_xlabel('Section', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' '), fontsize=12)
    ax.set_title(f'Comparison: Section {target_section} vs Similar Sections', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    
    return fig


def create_section_report_figures(events_df: pd.DataFrame, 
                                  section_id: int,
                                  output_dir: str,
                                  risk_results: Optional[pd.DataFrame] = None,
                                  similar_sections: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Create all standard figures for a section analysis report.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Section events
    section_id : int
        Section ID
    output_dir : str
        Directory to save figures
    risk_results : pd.DataFrame, optional
        Risk scoring results
    similar_sections : pd.DataFrame, optional
        Similar sections for comparison
        
    Returns
    -------
    List[str]
        Paths to created figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    path = output_dir / f'fig1_section{section_id}_event_timeline.png'
    plot_event_timeline(events_df, section_id, str(path))
    created_files.append(str(path))
    plt.close()
    
    try:
        path = output_dir / f'fig2_section{section_id}_cause_distribution.png'
        plot_cause_distribution(events_df, section_id, str(path))
        created_files.append(str(path))
        plt.close()
    except ValueError:
        pass  # No cause column
    
    try:
        path = output_dir / f'fig3_section{section_id}_temporal_heatmap.png'
        plot_temporal_heatmap(events_df, section_id, str(path))
        created_files.append(str(path))
        plt.close()
    except Exception:
        pass
    
    if risk_results is not None:
        path = output_dir / f'fig4_section{section_id}_risk_distribution.png'
        plot_risk_distribution(risk_results, str(path), highlight_sections=[section_id])
        created_files.append(str(path))
        plt.close()
    
    if similar_sections is not None and len(similar_sections) > 0:
        path = output_dir / f'fig5_section{section_id}_comparison.png'
        plot_comparative_bar(section_id, similar_sections, 'Event_Count', str(path))
        created_files.append(str(path))
        plt.close()
    
    return created_files
