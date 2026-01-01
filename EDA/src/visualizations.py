"""
Visualization utilities for PMU disturbance analysis.
Supports both static (matplotlib/seaborn) and interactive (plotly) visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import config


def save_figure(fig, filename: str, static: bool = True, interactive: bool = True):
    """
    Save figure in both static and interactive formats.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    static : bool
        Save static versions (PNG, PDF)
    interactive : bool
        Save interactive version (HTML) if plotly figure
    """
    # Ensure output directories exist
    Path(config.FIGURE_DIR, 'static').mkdir(parents=True, exist_ok=True)
    Path(config.FIGURE_DIR, 'interactive').mkdir(parents=True, exist_ok=True)

    # Check if it's a plotly figure
    is_plotly = isinstance(fig, (go.Figure, go.FigureWidget))

    if is_plotly:
        # Save interactive HTML
        if interactive:
            html_path = Path(config.FIGURE_DIR, 'interactive', f'{filename}.html')
            fig.write_html(str(html_path))

        # Save static versions
        if static:
            png_path = Path(config.FIGURE_DIR, 'static', f'{filename}.png')
            pdf_path = Path(config.FIGURE_DIR, 'static', f'{filename}.pdf')
            fig.write_image(str(png_path), width=1200, height=800, scale=2)
            fig.write_image(str(pdf_path), width=1200, height=800)
    else:
        # Matplotlib figure
        if static:
            png_path = Path(config.FIGURE_DIR, 'static', f'{filename}.png')
            pdf_path = Path(config.FIGURE_DIR, 'static', f'{filename}.pdf')
            fig.savefig(png_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
            fig.savefig(pdf_path, dpi=config.FIGURE_DPI, bbox_inches='tight')


def plot_time_series_decomposition(decomposition: Dict,
                                   title: str = 'Time Series Decomposition',
                                   interactive: bool = False):
    """
    Plot STL decomposition results.

    Parameters:
    -----------
    decomposition : Dict
        Dictionary with 'observed', 'trend', 'seasonal', 'residual'
    title : str
        Plot title
    interactive : bool
        Use plotly for interactive plot
    """
    if interactive:
        # Plotly version
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )

        components = ['observed', 'trend', 'seasonal', 'residual']
        for i, comp in enumerate(components, 1):
            fig.add_trace(
                go.Scatter(x=decomposition[comp].index, y=decomposition[comp].values,
                          mode='lines', name=comp.capitalize()),
                row=i, col=1
            )

        fig.update_layout(height=800, title_text=title, showlegend=False)
        return fig
    else:
        # Matplotlib version
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        decomposition['observed'].plot(ax=axes[0], title='Observed')
        decomposition['trend'].plot(ax=axes[1], title='Trend')
        decomposition['seasonal'].plot(ax=axes[2], title='Seasonal')
        decomposition['residual'].plot(ax=axes[3], title='Residual')

        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout()
        return fig


def plot_anomalies(ts: pd.Series,
                   anomalies: Dict[str, pd.Series],
                   title: str = 'Anomaly Detection',
                   interactive: bool = False):
    """
    Plot time series with detected anomalies from multiple methods.

    Parameters:
    -----------
    ts : pd.Series
        Original time series
    anomalies : Dict[str, pd.Series]
        Dictionary of anomaly detection results (method_name: boolean series)
    title : str
        Plot title
    interactive : bool
        Use plotly for interactive plot
    """
    if interactive:
        fig = go.Figure()

        # Plot original series
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines',
                                name='Disturbances', line=dict(color='blue')))

        # Plot anomalies from each method
        colors = ['red', 'orange', 'purple']
        for i, (method, anom_series) in enumerate(anomalies.items()):
            anomaly_idx = ts.index[anom_series]
            fig.add_trace(go.Scatter(x=anomaly_idx, y=ts[anom_series],
                                    mode='markers', name=f'Anomalies ({method})',
                                    marker=dict(color=colors[i % len(colors)], size=10,
                                              symbol='circle-open', line=dict(width=2))))

        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Count',
                         height=500, hovermode='x unified')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot original series
        ax.plot(ts.index, ts.values, label='Disturbances', color='blue', linewidth=1.5)

        # Plot anomalies from each method
        colors = ['red', 'orange', 'purple']
        markers = ['o', 's', '^']
        for i, (method, anom_series) in enumerate(anomalies.items()):
            anomaly_idx = ts.index[anom_series]
            ax.scatter(anomaly_idx, ts[anom_series], label=f'Anomalies ({method})',
                      color=colors[i % len(colors)], marker=markers[i % len(markers)],
                      s=100, alpha=0.7, edgecolors='black', linewidths=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig


def plot_calendar_heatmap(df: pd.DataFrame,
                          datetime_col: str,
                          title: str = 'Disturbance Calendar Heatmap'):
    """
    Create calendar heatmap using plotly (interactive only).

    Parameters:
    -----------
    df : pd.DataFrame
        Disturbance data
    datetime_col : str
        Datetime column name
    title : str
        Plot title
    """
    # Aggregate by date
    df = df.copy()
    df['date'] = pd.to_datetime(df[datetime_col]).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])

    # Create figure
    fig = px.density_heatmap(
        daily_counts,
        x=daily_counts['date'].dt.month,
        y=daily_counts['date'].dt.day,
        z='count',
        color_continuous_scale='YlOrRd',
        labels={'x': 'Month', 'y': 'Day of Month', 'z': 'Disturbances'}
    )

    fig.update_layout(title=title, height=600)
    return fig


def plot_rolling_statistics(rolling_stats: pd.DataFrame,
                            title: str = 'Rolling Statistics',
                            interactive: bool = False):
    """
    Plot rolling mean and standard deviation.

    Parameters:
    -----------
    rolling_stats : pd.DataFrame
        DataFrame with rolling statistics
    title : str
        Plot title
    interactive : bool
        Use plotly for interactive plot
    """
    if interactive:
        fig = go.Figure()

        # Plot original
        fig.add_trace(go.Scatter(x=rolling_stats.index, y=rolling_stats['original'],
                                mode='lines', name='Original', opacity=0.5))

        # Plot rolling means
        mean_cols = [col for col in rolling_stats.columns if 'rolling_mean' in col]
        for col in mean_cols:
            window = col.split('_')[-1]
            fig.add_trace(go.Scatter(x=rolling_stats.index, y=rolling_stats[col],
                                    mode='lines', name=f'Mean {window}'))

        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Count',
                         height=500, hovermode='x unified')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot original
        ax.plot(rolling_stats.index, rolling_stats['original'],
               label='Original', alpha=0.5, linewidth=1)

        # Plot rolling means
        mean_cols = [col for col in rolling_stats.columns if 'rolling_mean' in col]
        for col in mean_cols:
            window = col.split('_')[-1]
            ax.plot(rolling_stats.index, rolling_stats[col],
                   label=f'Mean {window}', linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Count')
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig


def plot_acf_pacf(acf_values: np.ndarray,
                 pacf_values: np.ndarray,
                 title: str = 'ACF and PACF'):
    """
    Plot Autocorrelation and Partial Autocorrelation functions.

    Parameters:
    -----------
    acf_values : np.ndarray
        ACF values
    pacf_values : np.ndarray
        PACF values
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot ACF
    ax1.stem(range(len(acf_values)), acf_values, basefmt=' ')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=1.96/np.sqrt(len(acf_values)), color='red', linestyle='--', linewidth=1)
    ax1.axhline(y=-1.96/np.sqrt(len(acf_values)), color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')
    ax1.set_title('Autocorrelation Function')
    ax1.grid(alpha=0.3)

    # Plot PACF
    ax2.stem(range(len(pacf_values)), pacf_values, basefmt=' ')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=1.96/np.sqrt(len(pacf_values)), color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=-1.96/np.sqrt(len(pacf_values)), color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('PACF')
    ax2.set_title('Partial Autocorrelation Function')
    ax2.grid(alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_cyclical_patterns(patterns: Dict,
                           title: str = 'Cyclical Patterns',
                           interactive: bool = False):
    """
    Plot hourly, daily, and monthly patterns.

    Parameters:
    -----------
    patterns : Dict
        Dictionary with 'hourly', 'daily', 'monthly' patterns
    title : str
        Plot title
    interactive : bool
        Use plotly for interactive plot
    """
    if interactive:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hourly Pattern', 'Daily Pattern', 'Monthly Pattern', 'Weekend vs Weekday'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Hourly
        fig.add_trace(go.Bar(x=patterns['hourly'].index, y=patterns['hourly'].values,
                            name='Hourly'), row=1, col=1)

        # Daily
        fig.add_trace(go.Bar(x=patterns['daily'].index, y=patterns['daily'].values,
                            name='Daily'), row=1, col=2)

        # Monthly
        fig.add_trace(go.Bar(x=patterns['monthly'].index, y=patterns['monthly'].values,
                            name='Monthly'), row=2, col=1)

        # Weekend vs Weekday
        weekend_labels = ['Weekday', 'Weekend']
        fig.add_trace(go.Bar(x=weekend_labels, y=patterns['weekend_vs_weekday'].values,
                            name='Weekend vs Weekday'), row=2, col=2)

        fig.update_layout(height=800, title_text=title, showlegend=False)
        return fig
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Hourly
        axes[0, 0].bar(patterns['hourly'].index, patterns['hourly'].values)
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Hourly Pattern')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Daily
        axes[0, 1].bar(range(len(patterns['daily'])), patterns['daily'].values)
        axes[0, 1].set_xticks(range(len(patterns['daily'])))
        axes[0, 1].set_xticklabels(patterns['daily'].index, rotation=45)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Daily Pattern')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Monthly
        axes[1, 0].bar(range(len(patterns['monthly'])), patterns['monthly'].values)
        axes[1, 0].set_xticks(range(len(patterns['monthly'])))
        axes[1, 0].set_xticklabels(patterns['monthly'].index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Monthly Pattern')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Weekend vs Weekday
        weekend_labels = ['Weekday', 'Weekend']
        axes[1, 1].bar(weekend_labels, patterns['weekend_vs_weekday'].values)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Weekend vs Weekday')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout()
        return fig
