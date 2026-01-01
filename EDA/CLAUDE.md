# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis project for power grid PMU (Phasor Measurement Unit) disturbance monitoring. The goal is to analyze historical disturbance data to identify patterns, predict failures, and provide actionable insights for grid operators.

**Data Source**: `data/PMU_disturbance.xlsx` contains:
- **PMUs sheet**: 533 PMU installations with location, voltage, type, and service dates
- **Disturbances sheet**: 9,369 disturbance events with timestamps, locations, causes, and operational data
- **Link**: SectionID connects PMU infrastructure to disturbance events

## Key Analysis Requirements

Refer to `EDA/PMU_disturbance_EDA.md` for the complete analysis specification. The analysis focuses on seven major areas:

1. **Advanced Temporal Analysis**: Time series decomposition, anomaly detection, change point detection, rolling statistics
2. **Disturbance Causality & Pattern Mining**: Cause analysis, sequential patterns, reliability metrics, root cause analysis
3. **Spatial & Network Analysis**: Geographic clustering, spatial heatmaps, network topology, voltage level analysis
4. **Predictive & Risk Modeling**: Risk scoring, survival analysis, time series forecasting, classification models
5. **Operational & Maintenance Insights**: PMU age analysis, service patterns, operations data analysis
6. **Statistical Validation**: Distribution testing, hypothesis testing, correlation analysis, causality testing
7. **Advanced Visualizations**: Publication-quality plots for temporal, spatial, statistical, and comparative analysis

## Required Python Libraries

When setting up the analysis environment, install:
- **Core**: pandas, numpy
- **Statistics**: statsmodels, scipy
- **Machine Learning**: scikit-learn
- **Network Analysis**: networkx
- **Time Series**: ruptures (change point detection)
- **Survival Analysis**: lifelines
- **Association Rules**: mlxtend
- **Visualization**: seaborn, matplotlib, plotly

## Expected Deliverables

1. Executive summary with top 10 critical insights and high-priority sections
2. Technical analysis report with methodology and statistical validation
3. Predictive models and risk scores for all 533 PMU sections
4. Root cause analysis with recommendations
5. Visualization package (15-20 high-quality plots)
6. Actionable maintenance and monitoring recommendations
7. Data quality assessment

## Data Structure Notes

- SectionID is the key linking PMUs to disturbances
- InService/OutService dates determine PMU age and availability
- Operations field contains operational data requiring decoding/analysis
- Spatial coordinates enable geographic clustering and network analysis
- Temporal analysis requires careful handling of disturbance timestamps
