# Section 150 Technical Analysis Report
*Generated: 2025-12-31 23:19*

## PMU Section Details

| Attribute | Value |
|-----------|-------|
| TermID | 250 |
| Terminal | Fixico-Weleetka |
| Substation | Fixico |
| Scheme | 73063109 |
| Type | 1 |
| Voltage | 69000 |
| InService | 2014-04-23 00:00:00 |
| OutService | None |
| Latitude | 35.245444 |
| Longitude | -96.672538 |
| SectionID | 150 |
| PMU Type | SEL-421-3 |
| SPP Bus Name | FIXICO 2 |
| SPP Bus Number | 515077 |
| Age_Years | 11.690622861054072 |
| Age_Days | 4270 |
| Event_Count | 301 |

## Statistical Test Results

### Rate Significance Tests
- **Poisson Test**: statistic = N/A, p-value = 0.0
- **Z-Test**: z = 82.5647, p-value = 0.0
- **Conclusion**: Significantly higher

### Time-to-Failure Analysis
- **Section 150 MTBF**: 393.08 hours (16.38 days)
- **Network MTBF**: 4092.94 hours (170.54 days)
- **KS Test**: statistic = 0.4095516136979552, p-value = 2.477099550377746e-44
- **Conclusion**: Different distributions

### Temporal Clustering Analysis
- **Dispersion Index**: 1.5038 (>1 indicates clustering)
- **Is Poisson Process**: False
- **Number of Burst Days**: 154
- **Number of Change Points**: 0
- **Conclusion**: Clustered

## Cause Distribution Analysis
- **Chi-square statistic**: None
- **Chi-square p-value**: None
- **Distribution differs from network**: None

### Detailed Cause Analysis
| Cause | Sec150 Count | Sec150 % | Network % | Relative Risk | Chi2 p-value | Significant |
|-------|--------------|----------|-----------|---------------|--------------|-------------|
| Unknown | 51 | 16.94% | 12.81% | 1.322 | 0.04420976311869584 | True |
| Weather | 29 | 9.63% | 9.65% | 0.998 | 1.0 | False |
| Weather, excluding lightning | 25 | 8.31% | 6.08% | 1.367 | 0.14370213339109278 | False |
| Lightning | 18 | 5.98% | 4.47% | 1.339 | 0.26987525122075695 | False |
| Weather, excluding lightning - Severe weather in the area. Notified DCC, PSO & OMPA. | 5 | 1.66% | 0.00% | inf | 3.4940498344049205e-28 | True |

## Unique Characteristics of Section 150
- Rank #1 in event frequency with 301 events (network max: 301)
- Voltage level 69000: 9.8% of network
- PMU Type '1': 88.6% of network

## Methodology
### Statistical Tests Used
1. **Poisson Test**: Tests if event count exceeds expected under Poisson assumption
2. **Chi-square Test**: Tests if cause distribution differs from network
3. **Kolmogorov-Smirnov Test**: Compares inter-arrival time distributions
4. **Mann-Whitney U Test**: Non-parametric test for MTBF differences
5. **Change Point Detection (PELT)**: Identifies regime changes in event frequency

### Similarity Matching
Similar sections identified using cosine similarity on normalized PMU features 
(voltage level, PMU type, age, location) excluding event count to avoid data leakage.
