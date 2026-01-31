<p align="center">
  <h1 align="center">Retail Calendar Pattern Finder  </h1>
    <p align="center">
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-red" />
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-yellow" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-blue" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-brown" />
  <img src="https://img.shields.io/badge/Project-Complete-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-cyan" />
</p>

### Seasonality + Event-like Spike Detection (Data Analyst × Data Science Capstone)

This project turns raw retail transactions into a **calendar intelligence system**:

- **Seasonality**: Which days and months consistently perform better?
- **Calendar interactions**: Which **month × day-of-week** combinations are unusually strong/weak?
- **Spike detection**: Which days look like **events** (promotions, anomalies, shifts) after removing normal seasonality?
- **Spike explanation**: For each spike day, what drove it, **transactions**, **units**, **AOV**, or **category mix**?

It includes:
- A reproducible **Python pipeline** that generates metrics + figures + report artifacts  
- A **Streamlit dashboard** to explore seasonality and drill into spike days  
- Exportable outputs (`CSV`, `JSON`) designed for analyst workflows and portfolio review

---

## Table of Contents

- [What you get](#what-you-get)
- [Business questions answered](#business-questions-answered)
- [Dataset](#dataset)
  - [Data dictionary](#data-dictionary)
  - [Coverage note (critical)](#coverage-note-critical)
- [Methodology](#methodology)
  - [Daily rollups](#1-daily-rollups)
  - [Seasonality analysis](#2-seasonality-analysis)
  - [Baseline expected revenue](#3-baseline-expected-revenue)
  - [Spike detection](#4-spike-detection)
  - [Spike explanation](#5-spike-explanation)
- [Results & Figures](#results--figures)
  - [1) Day-of-week seasonality](#1-day-of-week-seasonality)
  - [2) Monthly revenue](#2-monthly-revenue)
  - [3) Month × day-of-week heatmap](#3-month--day-of-week-heatmap)
  - [4) Residual spike scatter](#4-residual-spike-scatter)
- [Outputs](#outputs)
- [How to run](#how-to-run)
  - [Install](#install)
  - [Run pipeline](#run-pipeline)
  - [Run dashboard](#run-dashboard)
- [Dashboard guide](#dashboard-guide)
- [Project structure](#project-structure)
- [Quality checks](#quality-checks)
- [Limitations & assumptions](#limitations--assumptions)
- [Improvements / roadmap](#improvements--roadmap)

---

## What you get

### Pipeline artifacts
- `outputs/daily_metrics.csv` | daily revenue / txns / units / AOV
- `outputs/weekly_metrics.csv` | weekly rollups + number of observed days
- `outputs/daily_scored.csv` | adds baseline expected revenue + residuals + spike scores
- `outputs/spike_days.csv` | top spike days ranked
- `outputs/spike_cards.json` | JSON “Spike Cards” with drivers and context
- `outputs/quality_summary.json` | sanity/quality stats

### Visuals (generated automatically)
Stored in `reports/figures/`:
- `dow_revenue.png`
- `month_revenue.png`
- `heatmap_month_dow.png`
- `spike_scatter.png`

### Dashboard
Streamlit app (`app/app.py`) with:
- overview KPIs
- seasonality visuals + heatmap
- spike explorer + drilldowns
- quality summary

---

## Business questions answered

### Seasonality (planning)
- Which **day-of-week** is consistently strongest for revenue?
- Which months are strongest/weakest within observed data?
- Is performance driven by:
  - **more transactions** (traffic)
  - **more units per day** (basket size)
  - **higher AOV** (ticket size)

### Spikes (event detection)
- Which days are unusually high/low **after accounting for seasonality**?
- Are spikes driven by:
  - **transactions spike** (demand surge / traffic)
  - **AOV spike** (higher-priced purchases)
  - **units spike** (bulk buying)
  - **category mix shift** (one category dominates)

---

## Dataset

Input file used by default:
- `data/raw/retail_sales_dataset.csv`

This project uses the following Kaggle dataset:

- **Retail Sales Dataset** by **Terence Katua**  
- Dataset URL: https://www.kaggle.com/datasets/terencekatua/retail-sales-dataset

**Thanks to the dataset author** for publishing the data and making it available for learning and analysis. 
All credit for the dataset goes to the original creator and Kaggle hosting platform.

### Data dictionary

| Column | Meaning | Notes |
|---|---|---|
| `Date` | Transaction date | Parsed to daily granularity |
| `Transaction ID` | Transaction identifier | Used to count transactions per day |
| `Customer ID` | Customer identifier | In this dataset each appears once (so no repeat-customer analysis) |
| `Gender` | Customer gender | Optional slice for analysis |
| `Age` | Customer age | Optional slice; can be bucketed |
| `Product Category` | Category label | Used for category mix analysis |
| `Quantity` | Units bought | Used for units/day and basket drivers |
| `Price per Unit` | Unit price | Used for pricing distribution |
| `Total Amount` | Revenue for that transaction | Typically `Quantity × Price per Unit` |

### Coverage note (critical)

This dataset contains **transaction-days that appear in the file**.

If a date is missing:
- it does **NOT** mean the store had zero sales  
- it means: **no record exists for that day in this dataset**

**Why this matters:**  
Monthly totals are labeled as **“Observed Transaction-Days”** so I don’t accidentally treat missing dates as true zero-sales days.

---

## Methodology

This section explains exactly what the pipeline does and why.

### 1) Daily rollups

I aggregate transaction-level data to a daily grain:

- **Revenue (daily)**  
  \[
  revenue_d = \sum_{t \in day(d)} total\_amount_t
  \]

- **Transactions (daily)**  
  \[
  txns_d = \#\{transaction\_id \text{ on day } d\}
  \]

- **Units (daily)**  
  \[
  units_d = \sum_{t \in day(d)} quantity_t
  \]

- **AOV (average order value)**  
  \[
  aov_d = \frac{revenue_d}{txns_d}
  \]
  (Implementation uses mean of transaction totals; with unique `Transaction ID` this aligns with the concept.)

I also create calendar fields:
- `dow` (day name)
- `month` (YYYY-MM)
- `week` (week period)

---

### 2) Seasonality analysis

I compute seasonality using *observed days*:

#### Day-of-week summary
For each day-of-week:
- average daily revenue
- median daily revenue (robust to outliers)
- average txns, units, AOV
- number of observed days (`n_days`), **important for confidence**

#### Monthly summary
For each month:
- total monthly revenue (observed)
- total txns, units (observed)
- average AOV
- number of observed days

#### Month × day-of-week heatmap
I compute:
- average daily revenue per `(month, dow)` cell  
This exposes interaction effects like:
> “Tuesdays are not always strong, but they may be strong in May.”

---

### 3) Baseline expected revenue

To detect true “events”, we need an **expected value** for each day (a seasonal baseline).

This repo uses a simple hierarchy:

1) Use **(month, dow)** mean if there are enough observations in that cell  
2) Otherwise fallback to **month** mean  
3) Otherwise fallback to **dow** mean  
4) Otherwise fallback to **overall** mean  

This is intentionally:
- explainable (good for analysts)
- stable for small datasets
- good enough for removing first-order seasonality

Formally, expected revenue for day \( d \) is:

\[
\hat{y}_d = E[revenue \mid month(d), dow(d)]
\]
with fallbacks as above when cell counts are too small.

---

### 4) Spike detection

I detect spikes with **two complementary detectors**:

#### A) Robust z-score on raw revenue (MAD-based)
This is great for quick, obvious outliers.

Given daily revenue series \( x \):
- median: \( m = median(x) \)
- MAD: \( mad = median(|x - m|) \)

Robust z-score:
\[
z = 0.6745 \cdot \frac{x - m}{mad}
\]

Why robust? Because retail revenue is often heavy-tailed and standard z-scores can be distorted by spikes.

#### B) Residual spike score vs expected baseline (recommended)
This isolates “event-like” behavior.

Residual:
\[
residual_d = revenue_d - \hat{y}_d
\]

I then compute a robust z-score on residuals:
\[
z^{resid}_d = robust\_z(residual_d)
\]

**Interpretation:**
- large positive residual = unusual surge beyond seasonality
- large negative residual = underperformance beyond seasonality

This is the method used to rank spike days in `outputs/spike_days.csv`.

---

### 5) Spike explanation

For the top-N spike days, I generate a JSON **Spike Card** that captures:

- actual revenue vs expected revenue
- residual and spike score
- driver snapshot:
  - transactions vs global baseline mean
  - units vs global baseline mean
  - AOV vs global baseline mean
- top category contribution and share
- notes about interpretation and coverage

This makes spike results:
- easy to paste into reports
- easy to use for dashboards
- easy to extend into “warning cards” later

---

## Results & Figures

All figures are generated by the pipeline and stored in `reports/figures/`.

### 1) Day-of-week seasonality

<img width="2000" height="800" alt="dow_revenue" src="https://github.com/user-attachments/assets/fcb56086-931b-4b26-afa3-c600e7e0e609" />

**What this chart shows**  
Average daily revenue for each day-of-week across observed days.

**How to interpret**
- Use this to identify the store’s weekly rhythm.
- If Saturday is highest, that suggests weekend demand strength.
- Use **median** alongside average for robustness (median isn’t shown in this figure, but is computed in outputs).

**Business use**
- staffing + labor planning
- inventory replenishment timing
- promotions on weaker weekdays

**Analyst caution**
- “Average revenue” can be inflated by a few spikes. That’s why the project also uses:
  - median stats
  - residual-based spike logic

---

### 2) Monthly revenue

<img width="2000" height="800" alt="month_revenue" src="https://github.com/user-attachments/assets/f8fc7da0-8d38-4c10-93bb-b94f45f7a10d" />

**What this chart shows**  
Total monthly revenue **only for transaction-days that exist in the dataset**.

**How to interpret**
- This is not guaranteed to represent full-month store revenue.
- If a month has fewer recorded days, it will appear lower.

**Best practice**
- Always pair monthly totals with:
  - number of observed days per month
  - whether the month is partial coverage (e.g., the last month may contain only one day)

**Business use**
- detect months with consistent uplift/downturn (within observed coverage)
- plan seasonal campaigns and stock strategies

---

### 3) Month × day-of-week heatmap

<img width="2000" height="1200" alt="heatmap_month_dow" src="https://github.com/user-attachments/assets/79a44d2c-3b19-420c-a731-5a8dd386b563" />

**What this chart shows**  
Average daily revenue by:
- month (rows)
- day-of-week (columns)

**Why it matters**  
This reveals calendar interactions that simple bar charts miss:
- “Tuesdays are average overall”  
  but  
- “Tuesdays in May are extremely strong”

**How to interpret**
- Bright cells = strong month+weekday combination
- Dark cells = weak combination
- Blank/white areas often mean **no observations** (coverage gaps)

**Business use**
- precision planning (“strong Tuesdays in May”)
- promo scheduling for specific calendar windows
- identify “unexpectedly strong” weekday blocks in specific months

---

### 4) Residual spike scatter

<img width="2000" height="800" alt="spike_scatter" src="https://github.com/user-attachments/assets/4a0b3571-af26-48b6-b80b-319c4c677823" />

**What this chart shows**  
For each day:
- residual = actual revenue − expected revenue

Expected revenue is computed from the seasonal hierarchy baseline (month+dow → month → dow → overall).

**How to interpret**
- Points near 0: normal days (seasonality explains them)
- Big positive spikes: event-like surges
- Big negative dips: event-like underperformance

**Why residuals are better than raw revenue**
Raw revenue spikes may simply be “normal Saturdays.”
Residuals answer:
> “Was this day abnormal compared to what the calendar predicts?”

**Business use**
- detect promotion impact
- detect data anomalies
- detect demand shocks

---

## Outputs

| Path | Purpose |
|---|---|
| `outputs/daily_metrics.csv` | Daily rollups (revenue, txns, units, AOV) |
| `outputs/weekly_metrics.csv` | Weekly rollups and observed-day counts |
| `outputs/daily_scored.csv` | Daily rollups + expected revenue + residuals + z-scores |
| `outputs/spike_days.csv` | Ranked spike days (top N) |
| `outputs/spike_cards.json` | Exportable spike explanations in JSON |
| `outputs/quality_summary.json` | Data quality stats |

---

## How to run

### Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
````

### Run pipeline

```bash
python -m src.pipeline --input data/raw/retail_sales_dataset.csv
```

After it finishes, check:

* `outputs/`
* `reports/figures/`
* `reports/insights.md`

### Run dashboard

```bash
streamlit run app/app.py
```

---

## Dashboard guide

### What you can do

* View KPI tiles (revenue, transactions, units, AOV)
* Explore seasonality:

  * DOW bar charts
  * monthly bars
  * month×DOW heatmap
* Spike explorer:

  * table of top spike days
  * choose a spike day → see drivers and category contribution
  * view/export Spike Card JSON
* Quality summary:

  * mismatches in `Total Amount = Quantity × Price per Unit`
  * negative/zero values, missing values

### How to interpret “Spike Explorer”

When you select a date:

* “Residual vs expected” tells you if it’s an event-like day
* Driver deltas tell you whether the spike came from:

  * more transactions (traffic)
  * higher AOV (ticket size)
  * more units (basket size)
* Category bar tells you whether one category dominated the day

---

## Project structure

```text
retail-calendar-pattern-finder/
  data/
    raw/                  # input CSV lives here
  src/
    io.py                 # loading + standardizing schema
    quality_checks.py     # sanity checks
    aggregate_daily.py    # daily metrics + category mix
    aggregate_weekly.py   # weekly metrics
    seasonality.py        # dow/month summaries + heatmap pivots
    baseline.py           # expected revenue model
    spike_detection.py    # robust z + residual z
    spike_explain.py      # spike cards JSON
    reporting.py          # markdown report generator
    pipeline.py           # CLI entrypoint
  app/
    app.py                # Streamlit dashboard
  outputs/                # generated CSV/JSON outputs
  reports/
    insights.md           # generated report
    figures/              # generated figures
```

---

## Quality checks

This repo includes sanity checks designed for real analyst workflows:

* Validate identity:

  * `Total Amount == Quantity × Price per Unit`
* Validate positivity:

  * quantities and prices should be > 0
* Date range profiling:

  * min/max date
* Missing values:

  * count rows with any missing values

Output:

* `outputs/quality_summary.json`

---

## Limitations & assumptions

### Dataset limitations

* **Coverage gaps**: missing dates mean “unknown coverage”
* **No store/region dimension**: cannot compare locations
* **No promotion/holiday labels**: spikes are statistically detected, not causally explained
* **Customer-level analysis is limited**: most customers appear once (no repeat behavior modeling)

### Modeling assumptions

* Expected revenue baseline is simple and explainable (good for analysts), but not perfect.
* Spikes are scored statistically; root cause requires business context or external signals.

---

## Improvements / roadmap

If you want to upgrade this project into a standout portfolio capstone:

### Better baseline

* Train a regression model for expected revenue using:

  * month, dow, week-of-year
  * category mix features
  * rolling averages
* Compare baseline accuracy and spike stability

### Better explanations

* Decompose revenue changes as:

  * **Revenue ≈ Transactions × AOV**
* Add “driver waterfall” for each spike day

### Calendar coverage support

* Create a complete calendar table and mark:

  * observed vs missing days
* Render heatmaps with both:

  * mean revenue
  * observation count overlay

### Production polish

* Add `Makefile` / `taskfile`
* Add formatting/linting (ruff/black)
* Add a GitHub Actions workflow to run pipeline checks
