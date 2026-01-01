"""
Predictive modeling utilities for PMU disturbance analysis.
Includes risk scoring, survival analysis, and time series forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_composite_risk_score(pmu_df: pd.DataFrame,
                                   disturbance_df: pd.DataFrame,
                                   section_col: str = 'SectionID',
                                   datetime_col: str = 'DateTime',
                                   weights: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate composite risk score for each PMU section.

    Parameters:
    -----------
    pmu_df : pd.DataFrame
        PMU installation data
    disturbance_df : pd.DataFrame
        Disturbance data
    section_col : str
        Section identifier column
    datetime_col : str
        Datetime column in disturbance data
    weights : Dict, optional
        Risk component weights (defaults to config values)

    Returns:
    --------
    pd.DataFrame
        Risk scores for all sections
    """
    import config

    if weights is None:
        weights = config.RISK_WEIGHTS

    # Initialize results
    sections = pmu_df[section_col].unique()
    results = []

    # Calculate disturbance frequency per section
    freq_by_section = disturbance_df.groupby(section_col).size()

    # Calculate time since last disturbance
    latest_disturbance = disturbance_df.groupby(section_col)[datetime_col].max()
    current_time = disturbance_df[datetime_col].max()

    # Calculate PMU age
    pmu_indexed = pmu_df.set_index(section_col)
    pmu_age = {}
    for section in pmu_indexed.index:
        try:
            in_service = pmu_indexed.loc[section, 'InService']
            if pd.notna(in_service):
                pmu_age[section] = (current_time - in_service).days / 365.25
            else:
                pmu_age[section] = 0
        except:
            pmu_age[section] = 0

    for section in sections:
        # Historical frequency
        freq = freq_by_section.get(section, 0)

        # Time since last disturbance (higher = higher risk)
        if section in latest_disturbance.index:
            days_since = (current_time - latest_disturbance[section]).days
            time_score = min(days_since / 365, 1.0)  # Normalize to [0, 1]
        else:
            time_score = 1.0  # No disturbances = uncertain risk

        # PMU age score
        age = pmu_age.get(section, 0)
        age_score = min(age / 20, 1.0)  # Normalize to [0, 1], assuming 20 years max

        # Placeholder for cause severity and trend (would be calculated from actual data)
        cause_severity_score = 0.5  # Default medium severity
        trend_score = 0.5  # Default neutral trend

        results.append({
            section_col: section,
            'Historical_Frequency': freq,
            'Frequency_Score': min(freq / freq_by_section.max(), 1.0) if len(freq_by_section) > 0 else 0,
            'Trend_Score': trend_score,
            'Cause_Severity_Score': cause_severity_score,
            'Time_Since_Last_Score': time_score,
            'Age_Score': age_score
        })

    risk_df = pd.DataFrame(results)

    # Calculate weighted composite score
    risk_df['Composite_Risk_Score'] = (
        risk_df['Frequency_Score'] * weights['historical_frequency'] +
        risk_df['Trend_Score'] * weights['trend_direction'] +
        risk_df['Cause_Severity_Score'] * weights['cause_severity'] +
        risk_df['Time_Since_Last_Score'] * weights['time_since_last'] +
        risk_df['Age_Score'] * weights['pmu_age']
    )

    # Scale to 0-100
    risk_df['Risk_Score_0_100'] = (risk_df['Composite_Risk_Score'] * 100).round(2)

    # Assign risk categories
    risk_df['Risk_Category'] = pd.cut(
        risk_df['Risk_Score_0_100'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    return risk_df.sort_values('Risk_Score_0_100', ascending=False)


def fit_kaplan_meier(df: pd.DataFrame,
                     duration_col: str,
                     event_col: str,
                     group_col: Optional[str] = None) -> Dict:
    """
    Fit Kaplan-Meier survival curves.

    Parameters:
    -----------
    df : pd.DataFrame
        Survival data
    duration_col : str
        Time to event column
    event_col : str
        Event occurred indicator (1 = event, 0 = censored)
    group_col : str, optional
        Column to stratify by

    Returns:
    --------
    Dict
        Fitted KM models and summary statistics
    """
    kmf = KaplanMeierFitter()

    if group_col is None:
        # Single curve
        kmf.fit(df[duration_col], df[event_col])
        return {
            'kmf': kmf,
            'median_survival': kmf.median_survival_time_,
            'survival_function': kmf.survival_function_
        }
    else:
        # Stratified curves
        results = {}
        for group in df[group_col].unique():
            mask = df[group_col] == group
            kmf_group = KaplanMeierFitter()
            kmf_group.fit(df.loc[mask, duration_col], df.loc[mask, event_col], label=str(group))
            results[group] = {
                'kmf': kmf_group,
                'median_survival': kmf_group.median_survival_time_,
                'survival_function': kmf_group.survival_function_
            }
        return results


def fit_cox_model(df: pd.DataFrame,
                 duration_col: str,
                 event_col: str,
                 covariates: list) -> Dict:
    """
    Fit Cox Proportional Hazards model.

    Parameters:
    -----------
    df : pd.DataFrame
        Survival data with covariates
    duration_col : str
        Time to event column
    event_col : str
        Event occurred indicator
    covariates : list
        List of covariate column names

    Returns:
    --------
    Dict
        Fitted Cox model and summary
    """
    cph = CoxPHFitter()

    # Prepare data
    model_data = df[[duration_col, event_col] + covariates].dropna()

    # Fit model
    cph.fit(model_data, duration_col=duration_col, event_col=event_col)

    return {
        'model': cph,
        'summary': cph.summary,
        'concordance_index': cph.concordance_index_,
        'coefficients': cph.params_
    }


def forecast_arima(ts: pd.Series,
                  order: Tuple[int, int, int] = (1, 1, 1),
                  forecast_periods: int = 30) -> Dict:
    """
    Forecast using ARIMA model.

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    order : Tuple[int, int, int]
        ARIMA order (p, d, q)
    forecast_periods : int
        Number of periods to forecast

    Returns:
    --------
    Dict
        Forecast results and diagnostics
    """
    # Fit ARIMA model
    model = ARIMA(ts, order=order)
    fitted_model = model.fit()

    # Generate forecast
    forecast = fitted_model.forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_periods + 1, freq=ts.index.freq)[1:]

    # Calculate confidence intervals
    forecast_result = fitted_model.get_forecast(steps=forecast_periods)
    forecast_ci = forecast_result.conf_int()

    return {
        'model': fitted_model,
        'forecast': pd.Series(forecast, index=forecast_index),
        'confidence_interval': forecast_ci,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'fitted_values': fitted_model.fittedvalues,
        'residuals': fitted_model.resid
    }


def forecast_sarima(ts: pd.Series,
                   order: Tuple[int, int, int] = (1, 1, 1),
                   seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
                   forecast_periods: int = 30) -> Dict:
    """
    Forecast using SARIMA model (seasonal ARIMA).

    Parameters:
    -----------
    ts : pd.Series
        Time series data
    order : Tuple[int, int, int]
        ARIMA order (p, d, q)
    seasonal_order : Tuple[int, int, int, int]
        Seasonal order (P, D, Q, s)
    forecast_periods : int
        Number of periods to forecast

    Returns:
    --------
    Dict
        Forecast results and diagnostics
    """
    # Fit SARIMA model
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit(disp=False)

    # Generate forecast
    forecast = fitted_model.forecast(steps=forecast_periods)
    forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_periods + 1, freq=ts.index.freq)[1:]

    # Calculate confidence intervals
    forecast_result = fitted_model.get_forecast(steps=forecast_periods)
    forecast_ci = forecast_result.conf_int()

    return {
        'model': fitted_model,
        'forecast': pd.Series(forecast, index=forecast_index),
        'confidence_interval': forecast_ci,
        'aic': fitted_model.aic,
        'bic': fitted_model.bic,
        'fitted_values': fitted_model.fittedvalues,
        'residuals': fitted_model.resid
    }


def evaluate_forecast(actual: pd.Series,
                     predicted: pd.Series) -> Dict:
    """
    Evaluate forecast accuracy.

    Parameters:
    -----------
    actual : pd.Series
        Actual values
    predicted : pd.Series
        Predicted values

    Returns:
    --------
    Dict
        Evaluation metrics
    """
    # Align series
    common_index = actual.index.intersection(predicted.index)
    actual_aligned = actual.loc[common_index]
    predicted_aligned = predicted.loc[common_index]

    # Calculate metrics
    mae = mean_absolute_error(actual_aligned, predicted_aligned)
    mse = mean_squared_error(actual_aligned, predicted_aligned)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'n_observations': len(common_index)
    }


def predict_section_disturbances(risk_scores: pd.DataFrame,
                                 historical_data: pd.DataFrame,
                                 forecast_days: int = 30) -> pd.DataFrame:
    """
    Predict expected disturbances per section for future period.

    Parameters:
    -----------
    risk_scores : pd.DataFrame
        Risk scores from calculate_composite_risk_score
    historical_data : pd.DataFrame
        Historical disturbance data
    forecast_days : int
        Forecast horizon in days

    Returns:
    --------
    pd.DataFrame
        Predicted disturbances with confidence intervals
    """
    predictions = []

    for idx, row in risk_scores.iterrows():
        section_id = row['SectionID']
        risk_score = row['Risk_Score_0_100']

        # Simple prediction based on historical frequency and risk score
        hist_freq = row['Historical_Frequency']

        # Assume linear relationship with risk score
        base_rate = hist_freq / 365  # daily rate
        adjusted_rate = base_rate * (risk_score / 50)  # Scale by risk

        # Predict for forecast period
        expected = adjusted_rate * forecast_days

        # Simple confidence interval (±50%)
        lower_bound = expected * 0.5
        upper_bound = expected * 1.5

        predictions.append({
            'SectionID': section_id,
            'Forecast_Days': forecast_days,
            'Expected_Disturbances': round(expected, 2),
            'Lower_Bound': round(lower_bound, 2),
            'Upper_Bound': round(upper_bound, 2),
            'Risk_Score': risk_score
        })

    return pd.DataFrame(predictions).sort_values('Expected_Disturbances', ascending=False)
