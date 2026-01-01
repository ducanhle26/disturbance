# PMU Reliability Framework - Implementation Plan for Claude Code CLI

---

## 🎯 Project Goal

Build a production-ready Python package that:
1. **Reproduces** all Section 150 analysis findings (301 events, 16.4 day MTBF, 25.1x ratio)
2. **Generalizes** to analyze ANY section in the network (not hardcoded to Section 150)
3. **Implements** novel multi-dimensional risk scoring framework
4. **Generates** publication-ready figures and reports
5. **Validates** with comprehensive tests to ensure reproducibility

---

## 📦 Project Structure
```
pmu_reliability_framework/
├── src/                    # Core modules
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Executable scripts
├── outputs/                # Generated figures, tables, reports
└── data/                   # Data directory
```

---

## 🔧 Implementation Tasks

### **Phase 1: Core Infrastructure (Priority 1)**

#### Task 1: Data Loading Module
**File**: `src/data_loader.py`

**Requirements:**
- Load PMU_disturbance.xlsx (both sheets)
- Validate data structure (533 PMUs, 9,369 events)
- Convert date columns to datetime
- Add derived columns (PMU age in years/days)
- Provide function to extract events for any section
- Calculate basic statistics (event count, MTBF, date range)

**Key Functions:**
```python
def load_pmu_disturbance_data(filepath) -> (pmu_df, dist_df)
def get_section_events(dist_df, section_id) -> section_events_df
def calculate_event_statistics(events_df) -> dict with {count, mtbf_days, first_event, last_event}
```

**Validation Criteria:**
- Section 150 returns exactly 301 events
- MTBF calculation returns ~16.4 days for Section 150
- All date columns are proper datetime objects

---

#### Task 2: Risk Scoring Module
**File**: `src/risk_scorer.py`

**Requirements:**
- Implement multi-dimensional risk scoring combining:
  - Frequency (event count)
  - Trend (are events increasing over time?)
  - MTBF (mean time between failures - inverted)
  - Age (equipment age)
  - Recency (time since last event - inverted)
- Default weights: frequency=35%, trend=25%, mtbf=20%, age=10%, recency=10%
- Normalize all components to 0-100 scale
- Calculate composite risk score (weighted sum)
- Rank all sections by risk
- Categorize as Low/Medium/High risk

**Key Functions:**
```python
class PMURiskScorer:
    def __init__(pmu_df, dist_df, weights=None)
    def calculate_risk_scores() -> results_df with columns:
        - SectionID
        - risk_score
        - rank
        - category
        - individual component scores
```

**Validation Criteria:**
- Section 150 ranks #1 (highest risk)
- Risk score for Section 150 is ~60-70 range
- All scores are between 0-100
- Weights sum to 1.0

---

#### Task 3: Temporal Analysis Module
**File**: `src/temporal_analysis.py`

**Requirements:**
- Detect anomalous days (IQR or Z-score method)
- Calculate hourly/daily/monthly patterns
- Identify peak hour, peak day, peak month
- Perform time series decomposition (trend, seasonal, residual)
- Calculate inter-arrival times between events
- Test for temporal clustering (dispersion index)

**Key Functions:**
```python
class TemporalAnalyzer:
    def detect_anomalies(events, method='iqr') -> anomalous_days_df
    def calculate_hourly_pattern(events) -> hourly_counts
    def calculate_peak_periods(events) -> {peak_hour, peak_day, peak_month}
    def test_clustering(events) -> {dispersion_index, is_clustered}
```

**Validation Criteria:**
- Section 150 peak hour = 19:00 (7 PM)
- Network average peak hour = 10:00 (10 AM)
- Dispersion index > 1.0 indicates clustering

---

#### Task 4: Spatial Analysis Module
**File**: `src/spatial_analysis.py`

**Requirements:**
- Geographic clustering using DBSCAN on lat/lon coordinates
- Calculate disturbance density by geographic area
- Find similar sections using cosine similarity on features (voltage, type, age, location)
- Identify k most similar sections to a given section (excluding event count to avoid data leakage)

**Key Functions:**
```python
class SpatialAnalyzer:
    def cluster_geographic(pmu_df, eps=0.5) -> pmu_df with cluster_id column
    def find_similar_sections(pmu_df, dist_df, section_id, k=10) -> similar_sections_df
```

**Validation Criteria:**
- Identifies 5-10 geographic clusters
- Similar sections have same voltage and PMU type as target section
- Similarity score between 0-1

---

#### Task 5: Visualization Module
**File**: `src/visualization.py`

**Requirements:**
- Create all standard plots for section analysis:
  - Event timeline with trend line
  - Cause distribution (bar chart + pie chart)
  - Temporal patterns (hour/day/month heatmaps)
  - Risk score distribution
  - Geographic map with event counts
  - MTBF histogram
  - Comparative plots (section vs similar sections)
- Save publication-quality figures (300 DPI, proper sizing)
- Consistent color scheme and styling

**Key Functions:**
```python
def plot_event_timeline(events, section_id, save_path)
def plot_cause_distribution(events, section_id, save_path)
def plot_temporal_heatmap(events, section_id, save_path)
def plot_risk_distribution(risk_results, save_path)
def plot_geographic_map(pmu_df, dist_df, save_path)
def create_section_report_figures(section_id, output_dir) -> creates all 7-10 figures
```

**Validation Criteria:**
- All figures saved as PNG at 300 DPI
- Proper titles, labels, legends
- Consistent Seaborn styling

---

### **Phase 2: Validation & Testing (Priority 2)**

#### Task 6: Reproducibility Tests
**File**: `tests/test_reproducibility.py`

**Requirements:**
Create pytest tests that validate:
- Section 150 has exactly 301 events
- Section 150 MTBF = 16.38 ± 0.5 days
- Section 150 network ratio = 25.08 ± 1.0x
- Section 150 ranks #1 in risk scoring
- Top cause is "Unknown" with 51 events (16.9%)
- Peak hour is 19:00 for Section 150

**Test Functions:**
```python
def test_section_150_event_count()
def test_section_150_mtbf()
def test_section_150_network_ratio()
def test_section_150_risk_rank()
def test_section_150_top_cause()
def test_section_150_peak_hour()
```

**Success Criteria:**
- All tests pass with pytest
- No assertion errors
- Runtime < 10 seconds

---

#### Task 7: Module Unit Tests
**Files**: 
- `tests/test_data_loader.py`
- `tests/test_risk_scorer.py`
- `tests/test_temporal_analysis.py`

**Requirements:**
- Test each module independently
- Test edge cases (empty data, single event, missing columns)
- Verify data type correctness
- Check calculation accuracy

**Success Criteria:**
- 80%+ code coverage
- All edge cases handled gracefully
- Clear error messages for failures

---

### **Phase 3: Analysis Scripts (Priority 3)**

#### Task 8: Full Network Analysis Script
**File**: `scripts/run_full_analysis.py`

**Requirements:**
- Analyze ALL 533 sections in the network
- Calculate risk scores for all sections
- Identify top 20 highest-risk sections
- Generate summary statistics
- Save results to CSV

**Execution:**
```bash
python scripts/run_full_analysis.py
```

**Outputs:**
- `outputs/results/network_risk_scores.csv` (all 533 sections)
- `outputs/results/top_risk_sections.csv` (top 20)
- `outputs/reports/network_summary.txt` (summary statistics)

---

#### Task 9: Section-Specific Report Generator
**File**: `scripts/generate_section_report.py`

**Requirements:**
- Accept section_id as command-line argument
- Perform complete deep-dive analysis on that section
- Generate all visualizations
- Create markdown report with findings
- Compare to similar sections

**Execution:**
```bash
python scripts/generate_section_report.py --section_id 150
```

**Outputs:**
- `outputs/reports/section_150_report.md` (markdown report)
- `outputs/figures/section_reports/section_150/` (7-10 figures)
- Summary of findings, recommendations, comparative analysis

---

#### Task 10: Paper Figures Generator
**File**: `scripts/create_paper_figures.py`

**Requirements:**
- Generate ALL figures needed for research paper
- Publication quality (high resolution, proper fonts, clear labels)
- Figures should include:
  - Network-wide risk distribution
  - Geographic clustering map
  - Temporal pattern comparison (Section 150 vs network)
  - Risk scoring methodology flowchart
  - Validation results (model performance)
  - Top 10 risk sections comparison

**Execution:**
```bash
python scripts/create_paper_figures.py
```

**Outputs:**
- `outputs/figures/publication/` directory with 10-15 figures
- Figure captions in `outputs/figures/publication/figure_captions.md`

---

### **Phase 4: Documentation & Notebooks (Priority 4)**

#### Task 11: Jupyter Notebooks
**Files**:
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_section_150_validation.ipynb`
- `notebooks/03_risk_scoring_framework.ipynb`
- `notebooks/04_framework_generalization.ipynb`

**Requirements:**
- Interactive exploration and validation
- Step-by-step walkthroughs with explanations
- Visualizations inline
- Can be run cell-by-cell for understanding

---

#### Task 12: Documentation
**Files**:
- `README.md` (installation, quickstart, usage examples)
- `docs/methodology.md` (detailed explanation of risk scoring methodology)
- `docs/api_reference.md` (function/class documentation)

**Requirements:**
- Clear installation instructions
- Example usage for common tasks
- Explanation of risk scoring formula
- Citation information for research paper

---

## ✅ Validation Checklist

Before considering the implementation complete, verify:

### Code Quality
- [ ] All modules have docstrings
- [ ] Functions have type hints
- [ ] Code follows PEP 8 style
- [ ] No hardcoded paths or values
- [ ] Error handling for edge cases

### Reproducibility
- [ ] All tests pass (`pytest tests/`)
- [ ] Section 150 metrics match original analysis
- [ ] Results are deterministic (same output every run)
- [ ] Can run on different machines

### Generalizability
- [ ] Works for ANY section_id (not just 150)
- [ ] Handles sections with 0 events gracefully
- [ ] Handles sections with 1 event gracefully
- [ ] Scales to full network (533 sections)

### Research Quality
- [ ] Risk scoring methodology is clearly documented
- [ ] Statistical validation included
- [ ] Publication-ready figures generated
- [ ] Results can be cited in paper

---

## 🎯 Success Metrics

**Phase 1 Complete When:**
- Can load data and calculate Section 150 MTBF = 16.4 days ✅
- Risk scorer ranks Section 150 as #1 ✅
- All 5 core modules implemented ✅

**Phase 2 Complete When:**
- All reproducibility tests pass ✅
- 80%+ code coverage ✅
- No failing tests ✅

**Phase 3 Complete When:**
- Full network analysis runs successfully ✅
- Section report generator works for any section ✅
- Paper figures are publication-ready ✅

**Phase 4 Complete When:**
- Documentation is complete and clear ✅
- Notebooks run without errors ✅
- README has usage examples ✅

---

## 📝 Implementation Notes

**For Claude Code CLI:**
- Use plan mode for systematic execution: `/plan`
- Prioritize Phase 1 first (core functionality)
- Validate each module before moving to next
- Run tests frequently to catch issues early
- Generate outputs in `outputs/` directory (automatically shared)
- Focus on clean, readable, well-documented code
- Avoid over-engineering - keep it simple and functional

**Key Principles:**
1. **Modular**: Each module does one thing well
2. **Testable**: Every function can be tested independently
3. **Reusable**: Works for any section, not just Section 150
4. **Reproducible**: Same inputs → same outputs always
5. **Documented**: Code is self-explanatory with clear docstrings

---

## 🚀 Expected Timeline

- **Phase 1**: 2-3 hours (core modules)
- **Phase 2**: 1-2 hours (testing)
- **Phase 3**: 1-2 hours (scripts and analysis)
- **Phase 4**: 1 hour (documentation and notebooks)

**Total**: 5-8 hours of focused development

---

## 📊 Final Deliverables

1. **Python Package**: `pmu_reliability_framework/` with all modules
2. **Test Suite**: Complete pytest suite with 80%+ coverage
3. **Analysis Results**: Risk scores for all 533 sections
4. **Section 150 Report**: Validated deep-dive matching original analysis
5. **Publication Figures**: 10-15 high-quality figures for paper
6. **Documentation**: README, API reference, methodology guide
7. **Notebooks**: Interactive exploration and validation

All code should be ready to:
- Run on any machine with Python 3.9+
- Share on GitHub for research paper submission
- Extend for future analysis (weather integration, ML models)
- Serve as foundation for publication

---

**End of Implementation Plan**
```

This plan is:
- ✅ **Clear**: Specific tasks with exact requirements
- ✅ **Actionable**: Claude Code CLI knows exactly what to build
- ✅ **Testable**: Each phase has validation criteria
- ✅ **Comprehensive**: Covers everything needed for research publication
- ✅ **Concise**: No unnecessary details, just what matters

Save this as `IMPLEMENTATION_PLAN.md` and give it to Claude Code CLI with:
```
/plan Execute the complete implementation plan in IMPLEMENTATION_PLAN.md