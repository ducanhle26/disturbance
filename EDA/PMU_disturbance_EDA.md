# Advanced PMU Disturbance Data Analysis

## Dataset Context
Analyze PMU_disturbance.xlsx containing power grid monitoring data:
- **PMUs sheet**: 533 Phasor Measurement Unit installations with location, voltage, type, and service dates
- **Disturbances sheet**: 9,369 disturbance events with timestamps, locations, causes, and operational data
- **Link**: SectionID connects PMU infrastructure to disturbance events

## Analysis Objectives

### 1. ADVANCED TEMPORAL ANALYSIS
- **Time Series Decomposition**: Apply STL (Seasonal-Trend decomposition using LOESS) or classical decomposition to identify:
  - Long-term trends in disturbance frequency
  - Seasonal patterns (monthly, weekly, daily cycles)
  - Irregular/random components
- **Anomaly Detection**: Use statistical methods (Z-score, IQR, Isolation Forest) to identify unusual disturbance spikes
- **Inter-Arrival Time Analysis**: 
  - Calculate time between consecutive disturbances per section
  - Test if follows exponential distribution (Poisson process)
  - Identify sections with clustering vs random patterns
- **Change Point Detection**: Apply PELT or binary segmentation to find regime shifts in disturbance rates
- **Temporal Clustering Analysis**:
  - Do disturbances occur in bursts or cascades?
  - Calculate auto-correlation functions (ACF/PACF)
  - Identify time windows with abnormally high activity
- **Rolling Statistics**: 
  - 7-day, 30-day, 90-day moving averages and standard deviations
  - Identify accelerating or decelerating trends
- **Cyclical Patterns**:
  - Hour-of-day analysis (load-related patterns)
  - Day-of-week effects
  - Monthly/seasonal variations
  - Weekend vs weekday comparisons

### 2. DISTURBANCE CAUSALITY & PATTERN MINING
- **Cause Analysis**:
  - Frequency distribution and Pareto analysis of disturbance causes
  - Cause evolution over time (are certain causes increasing?)
  - Association rule mining between causes using Apriori algorithm
- **Co-occurrence Matrix**: Which causes tend to appear together or in sequence?
- **Sequential Pattern Mining**:
  - Does cause A frequently lead to cause B within a time window?
  - Build transition probability matrices
  - Identify cascade patterns
- **Section Reliability Metrics**:
  - Mean Time Between Failures (MTBF) per section
  - Mean Time To Repair (MTTR) if applicable from Operations data
  - Failure rate (λ) calculations
  - Reliability trends - improving or degrading sections
- **Root Cause Analysis**:
  - Statistical tests for cause-effect relationships
  - Which causes have the highest operational impact?

### 3. SPATIAL & NETWORK ANALYSIS
- **Geographic Clustering**:
  - Apply DBSCAN or K-means on PMU coordinates
  - Identify geographic zones with high disturbance density
  - Calculate disturbance rates per cluster
- **Spatial Heatmaps**:
  - Create kernel density estimation (KDE) plots of disturbances
  - Identify geographic hotspots
- **Spatial Autocorrelation**:
  - Moran's I or Geary's C statistics
  - Do disturbances in adjacent sections correlate?
  - Test for spatial propagation effects
- **Voltage Level Analysis**:
  - Disturbance rates by voltage class
  - Statistical tests for voltage-level vulnerability
  - Interaction effects between voltage and location
- **Network Topology**:
  - Build PMU connectivity network based on proximity or shared disturbances
  - Calculate network centrality metrics (betweenness, closeness, eigenvector)
  - Identify critical/vulnerable nodes
  - Community detection in disturbance patterns

### 4. PREDICTIVE & RISK MODELING
- **Risk Scoring System**:
  - Develop composite risk score for each PMU section based on:
    - Historical disturbance frequency
    - Trend direction (increasing/decreasing)
    - Cause severity
    - Time since last disturbance
    - Age of PMU equipment
  - Rank sections by risk level
- **Leading Indicators**:
  - Identify early warning signals preceding high-disturbance periods
  - Feature engineering from temporal patterns
  - Correlation with external factors if available
- **Probability Models**:
  - Build Poisson/Negative Binomial regression for disturbance counts
  - Predict expected disturbances by section/time/cause
  - Calculate prediction intervals
- **Survival Analysis**:
  - Kaplan-Meier survival curves for time-to-next-failure
  - Cox proportional hazards model
  - Identify factors affecting failure rates
- **Classification Models**:
  - Predict disturbance cause from section/time features
  - Identify which sections will experience disturbances next month
- **Time Series Forecasting**:
  - ARIMA/SARIMA models for disturbance count prediction
  - Prophet for capturing multiple seasonality
  - Forecast next 30-90 days of disturbance activity

### 5. OPERATIONAL & MAINTENANCE INSIGHTS
- **PMU Age Analysis**:
  - Calculate PMU age from InService dates
  - Correlate age with disturbance frequency (bathtub curve)
  - Identify aging equipment needing replacement
- **Service Pattern Analysis**:
  - Analyze OutService patterns and durations
  - Planned vs unplanned outages
  - Maintenance effectiveness assessment
- **Operations Data Decoding**:
  - Statistical analysis of Operations values
  - Correlation with disturbance severity/duration
  - Identify operational threshold violations
- **PMU Type Comparison**:
  - Reliability comparison across different PMU types
  - Statistical tests for performance differences
  - Cost-benefit if certain types are more reliable
- **Capacity & Stress Indicators**:
  - Sections operating near capacity
  - Stress metrics based on disturbance frequency and severity
  - Identify overloaded sections

### 6. STATISTICAL VALIDATION & HYPOTHESIS TESTING
- **Distribution Testing**:
  - Test if disturbances follow Poisson, Negative Binomial, or other distributions
  - Goodness-of-fit tests (Chi-square, Kolmogorov-Smirnov)
  - QQ-plots for visual validation
- **Hypothesis Tests**:
  - Are disturbance rates significantly different across voltage levels?
  - Do certain PMU types have statistically lower failure rates?
  - Has disturbance frequency changed over time? (Mann-Kendall trend test)
  - Are geographic clusters statistically significant?
- **Correlation Analysis**:
  - Spearman/Pearson correlations between all relevant variables
  - Partial correlations controlling for confounders
  - Correlation matrices with significance testing
- **Causality Testing**:
  - Granger causality: do disturbances in section A predict section B?
  - Cross-correlation functions for lagged relationships
- **Confidence Intervals**:
  - Bootstrap confidence intervals for all key metrics
  - Uncertainty quantification in predictions
- **Multiple Testing Correction**:
  - Apply Bonferroni or FDR correction where appropriate

### 7. ADVANCED VISUALIZATIONS
Create professional, publication-quality visualizations:
- **Temporal**:
  - Multi-panel time series plots with trend decomposition
  - Ridgeline plots for disturbance distributions over time
  - Calendar heatmaps showing disturbance patterns
  - Waterfall charts for cumulative effects
- **Spatial**:
  - Interactive geospatial maps with disturbance overlays
  - Choropleth maps of disturbance density
  - Network graphs showing PMU connections and criticality
  - 3D surface plots of spatiotemporal patterns
- **Statistical**:
  - Correlation matrices with hierarchical clustering
  - Survival curves by section/type
  - ROC curves for predictive models
  - Forest plots for effect sizes
- **Comparative**:
  - Violin/box plots for distributions across categories
  - Sankey diagrams for disturbance cause flows
  - Parallel coordinates for multivariate PMU profiles
  - Radar charts for section risk profiles

## Technical Requirements
- Use Python with advanced analytics libraries:
  - **pandas, numpy**: Data manipulation
  - **statsmodels**: Time series, statistical tests, regression
  - **scikit-learn**: ML models, clustering, anomaly detection
  - **scipy**: Statistical distributions, hypothesis tests
  - **networkx**: Network analysis
  - **seaborn, matplotlib, plotly**: Visualizations
  - **ruptures**: Change point detection
  - **lifelines**: Survival analysis
  - **mlxtend**: Association rules
- Apply rigorous statistical methods with proper validation
- Document assumptions and limitations
- Provide reproducible code with clear comments

## Deliverables

### 1. Executive Summary Report
- **Top 10 Critical Insights**: Data-driven findings with business impact
- **High-Priority Sections**: List of PMU sections requiring immediate attention with justification
- **Key Risk Factors**: Statistical evidence for major reliability drivers
- **Trends & Patterns**: Summary of temporal and spatial patterns discovered

### 2. Technical Analysis Report
- Detailed methodology for each analysis
- Statistical test results with p-values and effect sizes
- Model performance metrics and validation
- Assumptions and limitations clearly stated

### 3. Predictive Models & Scoring
- Risk scores for all 533 PMU sections (ranked list)
- Probability of disturbance by section for next 30/60/90 days
- Forecasts with confidence intervals
- Model validation metrics (RMSE, MAE, accuracy, AUC)

### 4. Root Cause Analysis
- Statistical breakdown of disturbance causes
- Causal relationships with evidence
- Recommendations for each major cause category
- Cost-benefit analysis of interventions if possible

### 5. Visualization Package
- Minimum 15-20 high-quality visualizations
- Dashboard-ready plots (PNG/PDF at 300 DPI)
- Interactive HTML plots where appropriate
- Geographic maps showing spatial patterns

### 6. Actionable Recommendations
- Prioritized list of sections for preventive maintenance
- Monitoring improvements based on findings
- Resource allocation guidance
- Early warning system design recommendations

### 7. Data Quality Assessment
- Missing data analysis and imputation strategies
- Outlier investigation with domain context
- Data consistency checks
- Recommendations for future data collection

## Success Criteria
The analysis should enable power grid operators to:
1. Predict which sections are most likely to experience disturbances
2. Understand root causes with statistical confidence
3. Prioritize maintenance and monitoring resources
4. Identify early warning signals of degradation
5. Make data-driven decisions on infrastructure investments

Focus on actionable insights backed by rigorous statistical analysis rather than just descriptive statistics.