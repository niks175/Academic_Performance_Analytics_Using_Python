"""
Academic Performance Analytics - Exploratory Data Analysis
==========================================================
EDA on the UCI "Student Performance" (Mathematics) dataset: 395 students,
33 attributes covering demographics, family background, study habits,
lifestyle and three period grades (G1, G2, G3; scale 0-20).

Usage:
    python analyse.py
    python analyse.py --input data/student-mat.csv --output-dir output --show
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PASS_MARK = 10  # G3 >= 10 counts as a pass (Portuguese 0-20 scale)

EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc",
    "Walc", "health", "absences", "G1", "G2", "G3",
]

sns.set_theme(style="whitegrid", context="notebook", palette="deep")


# --------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EDA on student performance data.")
    p.add_argument("--input", default="data/student-mat.csv", help="Input CSV path")
    p.add_argument("--output-dir", default="output", help="Folder for plots and report")
    p.add_argument("--show", action="store_true", help="Display plots interactively")
    return p.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read the CSV whether it is comma-separated, semicolon-separated, or the
    original malformed export (every row wrapped in quotes with doubled
    quotes inside)."""
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    delimiter = ";" if lines[0].count(";") > lines[0].count(",") else ","

    rows = []
    for ln in lines:
        if ln.startswith('"') and ln.endswith('"') and ln.count('""') > 0:
            ln = ln[1:-1].replace('""', '"')  # un-wrap malformed rows
        rows.append(next(csv.reader([ln], delimiter=delimiter, quotechar='"')))
    header, body = rows[0], rows[1:]
    return pd.DataFrame(body, columns=header)


def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"Error: file not found -> {filepath}")

    df = read_csv_robust(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        sys.exit(f"Error: missing expected column(s): {sorted(missing)}")

    # Convert numeric-looking columns to numbers, leave text columns alone
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().all():
            df[col] = converted
    return df


def clean_data(df: pd.DataFrame, log: list) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    log.append(f"Duplicate rows removed: {before - len(df)}")
    log.append(f"Missing values (total): {int(df.isnull().sum().sum())}")

    df = df.dropna().reset_index(drop=True)

    # Feature engineering
    df["avg_grade"] = df[["G1", "G2", "G3"]].mean(axis=1).round(2)
    df["grade_change"] = df["G3"] - df["G1"]
    df["passed"] = np.where(df["G3"] >= PASS_MARK, "Pass", "Fail")
    return df


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> list:
    out = []
    add = out.append
    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(exclude="number")

    add("=" * 60)
    add("DATASET OVERVIEW")
    add("=" * 60)
    add(f"Rows: {len(df)}   Columns: {df.shape[1]} (33 original + 3 engineered: avg_grade, grade_change, passed)")
    add(f"Numeric columns: {num.shape[1]}   Categorical columns: {cat.shape[1]}")

    add("\n" + "=" * 60)
    add("DESCRIPTIVE STATISTICS (key numeric columns)")
    add("=" * 60)
    add(df[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]]
        .describe().round(2).to_string())

    add("\n" + "=" * 60)
    add("KEY METRICS")
    add("=" * 60)
    add(f"Average final grade (G3):  {df['G3'].mean():.2f} / 20")
    add(f"Median final grade (G3):   {df['G3'].median():.1f}")
    add(f"Pass rate (G3 >= {PASS_MARK}):      {(df['passed'] == 'Pass').mean() * 100:.1f}%")
    add(f"Students with G3 = 0:      {(df['G3'] == 0).sum()}")
    add(f"Average absences:          {df['absences'].mean():.2f}")
    add(f"G1-G3 correlation:         {df['G1'].corr(df['G3']):.2f}")
    add(f"G2-G3 correlation:         {df['G2'].corr(df['G3']):.2f}")
    add(f"Absences-G3 correlation:   {df['absences'].corr(df['G3']):.2f}")

    add("\n" + "=" * 60)
    add("TOP CORRELATIONS WITH FINAL GRADE (G3)")
    add("=" * 60)
    corr = num.drop(columns=["avg_grade", "grade_change"]).corr()["G3"].drop("G3")
    add(corr.reindex(corr.abs().sort_values(ascending=False).index).head(8)
        .round(3).to_string())

    add("\n" + "=" * 60)
    add("AVERAGE G3 BY GROUP")
    add("=" * 60)
    for col in ["school", "sex", "address", "higher", "internet", "schoolsup", "romantic"]:
        means = df.groupby(col)["G3"].mean().round(2)
        add(f"{col:<10} " + "   ".join(f"{k}: {v}" for k, v in means.items()))
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def save(fig, out_dir: Path, name: str, show: bool):
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_grade_distribution(df, out_dir, show):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["G3"], bins=21, binrange=(0, 21), kde=True, ax=ax, color="#2a6fbb")
    ax.axvline(df["G3"].mean(), color="red", ls="--", label=f"Mean = {df['G3'].mean():.1f}")
    ax.axvline(PASS_MARK, color="green", ls=":", label=f"Pass mark = {PASS_MARK}")
    ax.set(title="Distribution of Final Grades (G3)", xlabel="Final grade (0-20)", ylabel="Students")
    ax.legend()
    save(fig, out_dir, "01_grade_distribution.png", show)


def plot_grade_progression(df, out_dir, show):
    fig, ax = plt.subplots(figsize=(7, 5))
    means = df[["G1", "G2", "G3"]].mean()
    bars = ax.bar(means.index, means.values, color=["#4c9be8", "#2a6fbb", "#1b4a80"])
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(title="Average Grade by Period", ylabel="Average grade (0-20)", ylim=(0, 14))
    save(fig, out_dir, "02_grade_progression.png", show)


def plot_correlation_heatmap(df, out_dir, show):
    cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
            "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap of Numeric Features")
    save(fig, out_dir, "03_correlation_heatmap.png", show)


def plot_top_correlations(df, out_dir, show):
    corr = (df.select_dtypes(include="number")
              .drop(columns=["avg_grade", "grade_change"]).corr()["G3"].drop("G3"))
    corr = corr.reindex(corr.abs().sort_values().index).tail(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(corr.index, corr.values, color=np.where(corr.values > 0, "#2a9d8f", "#e76f51"))
    ax.set(title="Features Most Correlated with Final Grade (G3)", xlabel="Pearson correlation")
    ax.axvline(0, color="black", lw=0.8)
    save(fig, out_dir, "04_top_correlations_with_G3.png", show)


def plot_absences(df, out_dir, show):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="absences", y="G3", alpha=0.6, s=50, ax=ax)
    sns.regplot(data=df, x="absences", y="G3", scatter=False, color="red", ax=ax)
    ax.set(title="Absences vs Final Grade", xlabel="Number of absences", ylabel="Final grade (G3)")
    save(fig, out_dir, "05_absences_vs_G3.png", show)


def plot_study_and_failures(df, out_dir, show):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x="studytime", y="G3", hue="studytime", palette="Blues",
                legend=False, ax=axes[0])
    axes[0].set(title="Final Grade by Weekly Study Time",
                xlabel="Study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", ylabel="G3")
    sns.boxplot(data=df, x="failures", y="G3", hue="failures", palette="Reds",
                legend=False, ax=axes[1])
    axes[1].set(title="Final Grade by Past Class Failures",
                xlabel="Past failures", ylabel="G3")
    save(fig, out_dir, "06_studytime_and_failures.png", show)


def plot_categorical_comparison(df, out_dir, show):
    cols = ["school", "sex", "address", "higher", "internet", "schoolsup"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, cols):
        sns.boxplot(data=df, x=col, y="G3", hue=col, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"G3 by {col}")
    fig.suptitle("Final Grade Across Student Groups", fontsize=14, y=1.01)
    save(fig, out_dir, "07_group_comparison.png", show)


def plot_alcohol(df, out_dir, show):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [(axes[0], "Dalc", "Workday alcohol use"),
                           (axes[1], "Walc", "Weekend alcohol use")]:
        sns.barplot(data=df, x=col, y="G3", hue=col, palette="Oranges",
                    legend=False, errorbar=None, ax=ax)
        ax.set(title=f"Average G3 by {title}", xlabel="1 (very low) to 5 (very high)",
               ylabel="Average G3")
    save(fig, out_dir, "08_alcohol_vs_G3.png", show)


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    log = []
    df = clean_data(load_data(args.input), log)
    report = ["Academic Performance Analytics - EDA Report", ""] + log + [""] + summarize(df)
    text = "\n".join(report)
    print(text)
    (out_dir / "summary_report.txt").write_text(text, encoding="utf-8")

    print(f"\nGenerating visualizations -> '{out_dir}/'")
    for fn in (plot_grade_distribution, plot_grade_progression, plot_correlation_heatmap,
               plot_top_correlations, plot_absences, plot_study_and_failures,
               plot_categorical_comparison, plot_alcohol):
        fn(df, out_dir, args.show)
    print("Done.")


if __name__ == "__main__":
    main()
