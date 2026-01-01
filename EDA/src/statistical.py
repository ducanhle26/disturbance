"""
Statistical validation utilities for PMU disturbance analysis.
Includes hypothesis testing, distribution fitting, and correlation analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, kstest, shapiro
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def test_distribution_fit(data: pd.Series,
                         distributions: List[str] = ['poisson', 'nbinom', 'norm']) -> pd.DataFrame:
    """
    Test goodness of fit for multiple distributions.

    Parameters:
    -----------
    data : pd.Series
        Data to test
    distributions : List[str]
        List of distribution names to test

    Returns:
    --------
    pd.DataFrame
        Test results for each distribution
    """
    results = []

    data_clean = data.dropna()

    for dist_name in distributions:
        try:
            if dist_name == 'poisson':
                # Fit Poisson distribution
                lambda_param = data_clean.mean()
                # K-S test
                ks_stat, p_value = kstest(data_clean, lambda x: stats.poisson.cdf(x, lambda_param))

                results.append({
                    'Distribution': 'Poisson',
                    'Parameter': f'λ={lambda_param:.2f}',
                    'KS_Statistic': ks_stat,
                    'P_Value': p_value,
                    'Fits_Well': p_value > 0.05
                })

            elif dist_name == 'nbinom':
                # Fit Negative Binomial
                mean = data_clean.mean()
                var = data_clean.var()
                if var > mean:
                    p = mean / var
                    n = mean * p / (1 - p)

                    # K-S test (approximate)
                    ks_stat, p_value = kstest(data_clean, lambda x: stats.nbinom.cdf(x, n, p))

                    results.append({
                        'Distribution': 'Negative Binomial',
                        'Parameter': f'n={n:.2f}, p={p:.2f}',
                        'KS_Statistic': ks_stat,
                        'P_Value': p_value,
                        'Fits_Well': p_value > 0.05
                    })

            elif dist_name == 'norm':
                # Fit Normal distribution
                mean, std = data_clean.mean(), data_clean.std()
                ks_stat, p_value = kstest(data_clean, lambda x: stats.norm.cdf(x, mean, std))

                results.append({
                    'Distribution': 'Normal',
                    'Parameter': f'μ={mean:.2f}, σ={std:.2f}',
                    'KS_Statistic': ks_stat,
                    'P_Value': p_value,
                    'Fits_Well': p_value > 0.05
                })

        except Exception as e:
            results.append({
                'Distribution': dist_name,
                'Parameter': 'Error',
                'KS_Statistic': np.nan,
                'P_Value': np.nan,
                'Fits_Well': False,
                'Error': str(e)
            })

    return pd.DataFrame(results).sort_values('P_Value', ascending=False)


def mann_kendall_test(ts: pd.Series) -> Dict:
    """
    Perform Mann-Kendall trend test.

    Parameters:
    -----------
    ts : pd.Series
        Time series data

    Returns:
    --------
    Dict
        Test results including trend direction and significance
    """
    from scipy.stats import kendalltau

    data = ts.dropna().values
    n = len(data)

    if n < 3:
        return {'error': 'Insufficient data for Mann-Kendall test'}

    # Calculate S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            S += np.sign(data[j] - data[i])

    # Calculate variance
    var_S = n * (n - 1) * (2 * n + 5) / 18

    # Calculate Z statistic
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))

    # Determine trend
    if p_value < 0.05:
        if S > 0:
            trend = 'Increasing'
        else:
            trend = 'Decreasing'
    else:
        trend = 'No significant trend'

    return {
        'S_statistic': S,
        'Z_score': Z,
        'P_value': p_value,
        'Trend': trend,
        'Significant': p_value < 0.05
    }


def test_voltage_disturbance_relationship(df: pd.DataFrame,
                                         voltage_col: str,
                                         disturbance_count_col: str) -> Dict:
    """
    Test relationship between voltage levels and disturbance rates using ANOVA.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with voltage and disturbance information
    voltage_col : str
        Voltage level column
    disturbance_count_col : str
        Disturbance count column

    Returns:
    --------
    Dict
        ANOVA test results
    """
    # Group by voltage level
    voltage_groups = [group[disturbance_count_col].dropna().values
                     for name, group in df.groupby(voltage_col)]

    # Remove empty groups
    voltage_groups = [g for g in voltage_groups if len(g) > 0]

    if len(voltage_groups) < 2:
        return {'error': 'Need at least 2 voltage groups'}

    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*voltage_groups)

    return {
        'F_statistic': f_stat,
        'P_value': p_value,
        'Significant': p_value < 0.05,
        'Num_Groups': len(voltage_groups),
        'Interpretation': f'Voltage level {"significantly affects" if p_value < 0.05 else "does not significantly affect"} disturbance rates (p={p_value:.4f})'
    }


def correlation_analysis(df: pd.DataFrame,
                        columns: Optional[List[str]] = None,
                        method: str = 'pearson') -> pd.DataFrame:
    """
    Perform correlation analysis with significance testing.

    Parameters:
    -----------
    df : pd.DataFrame
        Data
    columns : List[str], optional
        Columns to include (default: all numeric)
    method : str
        Correlation method ('pearson', 'spearman')

    Returns:
    --------
    pd.DataFrame
        Correlation matrix with significance indicators
    """
    if columns is None:
        # Select numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)

    # Calculate p-values
    n = len(df)
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), columns=corr_matrix.columns, index=corr_matrix.index)

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j:
                if method == 'pearson':
                    _, p_val = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                else:
                    _, p_val = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                p_values.iloc[i, j] = p_val

    return corr_matrix, p_values


def granger_causality_test(ts1: pd.Series,
                           ts2: pd.Series,
                           max_lag: int = 5) -> Dict:
    """
    Test Granger causality: does ts1 help predict ts2?

    Parameters:
    -----------
    ts1 : pd.Series
        First time series (potential cause)
    ts2 : pd.Series
        Second time series (potential effect)
    max_lag : int
        Maximum lag to test

    Returns:
    --------
    Dict
        Test results for each lag
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Prepare data
    data = pd.concat([ts2, ts1], axis=1).dropna()

    if len(data) < 2 * max_lag:
        return {'error': 'Insufficient data for Granger causality test'}

    try:
        # Perform test
        results = grangercausalitytests(data, max_lag, verbose=False)

        # Extract p-values
        summary = {}
        for lag in range(1, max_lag + 1):
            f_test = results[lag][0]['ssr_ftest']
            summary[f'Lag_{lag}'] = {
                'F_statistic': f_test[0],
                'P_value': f_test[1],
                'Significant': f_test[1] < 0.05
            }

        return summary
    except Exception as e:
        return {'error': str(e)}


def bootstrap_confidence_interval(data: pd.Series,
                                  statistic: str = 'mean',
                                  n_bootstrap: int = 1000,
                                  confidence_level: float = 0.95) -> Dict:
    """
    Calculate bootstrap confidence intervals.

    Parameters:
    -----------
    data : pd.Series
        Data
    statistic : str
        Statistic to calculate ('mean', 'median', 'std')
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level

    Returns:
    --------
    Dict
        Point estimate and confidence interval
    """
    data_clean = data.dropna().values

    if len(data_clean) == 0:
        return {'error': 'No valid data'}

    # Choose statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    elif statistic == 'std':
        stat_func = np.std
    else:
        return {'error': f'Unknown statistic: {statistic}'}

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_clean, size=len(data_clean), replace=True)
        bootstrap_stats.append(stat_func(sample))

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)

    return {
        'Statistic': statistic,
        'Point_Estimate': stat_func(data_clean),
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper,
        'Confidence_Level': confidence_level,
        'N_Bootstrap': n_bootstrap
    }


def multiple_testing_correction(p_values: List[float],
                                method: str = 'fdr_bh',
                                alpha: float = 0.05) -> Dict:
    """
    Apply multiple testing correction.

    Parameters:
    -----------
    p_values : List[float]
        List of p-values
    method : str
        Correction method ('bonferroni', 'fdr_bh')
    alpha : float
        Family-wise error rate

    Returns:
    --------
    Dict
        Corrected results
    """
    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(p_values, alpha=alpha, method=method)

    return {
        'Original_P_Values': p_values,
        'Corrected_P_Values': p_corrected.tolist(),
        'Reject_Null': reject.tolist(),
        'Method': method,
        'Alpha': alpha,
        'Num_Significant': reject.sum()
    }


def chi_square_independence_test(df: pd.DataFrame,
                                 col1: str,
                                 col2: str) -> Dict:
    """
    Test independence between two categorical variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Data
    col1 : str
        First categorical column
    col2 : str
        Second categorical column

    Returns:
    --------
    Dict
        Chi-square test results
    """
    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    return {
        'Chi2_Statistic': chi2_stat,
        'P_Value': p_value,
        'Degrees_of_Freedom': dof,
        'Significant': p_value < 0.05,
        'Contingency_Table': contingency_table,
        'Expected_Frequencies': pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns),
        'Interpretation': f'{col1} and {col2} are {"" if p_value < 0.05 else "not "}significantly associated (p={p_value:.4f})'
    }
