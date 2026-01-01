# AGENTS.md

## Project Overview
Power grid PMU (Phasor Measurement Unit) disturbance analysis. Data: `data/PMU_disturbance.xlsx` (533 PMUs, 9,369 events). SectionID links PMUs to disturbances.

## Commands
```bash
source venv/bin/activate              # Activate virtual environment
python EDA/run_analysis.py            # Run full EDA pipeline
python Section_150/run_section150.py  # Run Section 150 analysis
pip install -r EDA/requirements.txt   # Install dependencies

# PMU Reliability Framework
cd PMU_reliability
pytest tests/ -v                      # Run all tests
python scripts/run_full_analysis.py   # Full network risk analysis
python scripts/generate_section_report.py --section_id 150  # Section report
python scripts/create_paper_figures.py  # Publication figures
```

## Architecture
- **EDA/**: Exploratory analysis (temporal, causality, spatial, predictive, statistical modules)
- **Section_150/**: Deep-dive on highest-risk section (8 analysis modules, outputs figures/reports)
- **PMU_reliability/**: Production-ready reliability framework (risk scoring, tests, scripts)
- **data/**: Source Excel with PMUs and Disturbances sheets
- **venv/**: Python virtual environment

## Code Style
- Python 3.10+, use pandas/numpy for data, matplotlib/seaborn/plotly for viz
- Imports: stdlib → third-party → local modules (use `sys.path.insert` for cross-module)
- Config in `config.py` or `config_section150.py` (paths, constants)
- Outputs go to `outputs/` subdirectories (figures/static, figures/interactive, reports, data)
- Use descriptive function names, docstrings for public functions, type hints
