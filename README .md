# Academic Performance Analytics Using Python

Exploratory Data Analysis (EDA) and visualization of student performance, built with **pandas, NumPy, Matplotlib and Seaborn**.

The dataset is the UCI *Student Performance* data (Mathematics): **395 students, 33 attributes** covering demographics, family background, study habits, lifestyle and three period grades (`G1`, `G2`, `G3`, scored 0-20).

## Project Structure

```
├── analyse.py            # EDA script: cleaning, statistics, 8 visualizations
├── data/
│   └── student-mat.csv   # Cleaned dataset (comma-separated)
├── output/               # Generated plots + summary_report.txt
├── requirements.txt
└── README.md
```

## What the Script Does

1. **Loads and validates** the data (checks all 33 expected columns, fixes numeric types).
2. **Cleans** it: duplicate and missing-value checks (none found in this dataset).
3. **Engineers features**: `avg_grade`, `grade_change` (G3 - G1) and `passed` (G3 >= 10).
4. **Summarizes**: descriptive statistics, key metrics, top correlations, group averages (saved to `output/summary_report.txt`).
5. **Visualizes**: 8 charts saved as PNG files.

## Getting Started

```bash
git clone https://github.com/niks175/Academic_Performance_Analytics_Using_Python.git
cd Academic_Performance_Analytics_Using_Python
pip install -r requirements.txt
python analyse.py
```

Options:

```bash
python analyse.py --input data/student-mat.csv --output-dir output --show
```

## Key Findings

| Metric | Value |
|---|---|
| Students | 395 |
| Average final grade (G3) | 10.42 / 20 |
| Pass rate (G3 >= 10) | 67.1% |
| Students with G3 = 0 | 38 |
| G2 vs G3 correlation | 0.90 |
| G1 vs G3 correlation | 0.80 |
| Past failures vs G3 correlation | -0.36 |
| Absences vs G3 correlation | 0.03 |

- **Earlier grades predict the final grade.** G2 and G1 correlate very strongly with G3 (0.90 and 0.80).
- **Past failures hurt the most** among non-grade features (-0.36).
- **Parental education helps modestly** (mother's education 0.22, father's 0.15).
- **Students who want higher education score far higher**: average G3 of 10.6 vs 6.8 for those who don't.
- **Urban students, internet users and students without a romantic relationship** have higher average grades.
- **Absences show almost no linear relationship with G3** here, partly because many G3 = 0 students have 0 recorded absences.
- Correlation is not causation; these are patterns in one school-year sample.

## Visualizations

| | |
|---|---|
| ![Grade distribution](output/01_grade_distribution.png) | ![Grade progression](output/02_grade_progression.png) |
| ![Correlation heatmap](output/03_correlation_heatmap.png) | ![Top correlations](output/04_top_correlations_with_G3.png) |
| ![Absences vs G3](output/05_absences_vs_G3.png) | ![Study time and failures](output/06_studytime_and_failures.png) |
| ![Group comparison](output/07_group_comparison.png) | ![Alcohol vs G3](output/08_alcohol_vs_G3.png) |

## Data Source

P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, 2008. UCI Machine Learning Repository: Student Performance.

## Contributing

Suggestions and improvements are welcome. Open an issue or submit a pull request.
