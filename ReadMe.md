# Backblaze Hard Drive RUL Prediction Project

## Project Goal

Build an end-to-end **Remaining Useful Life (RUL)** prediction pipeline for Backblaze hard drives using real quarterly failure data from 2024. The project aims to predict hard drive failures at multiple time horizons (2, 7, 14, and 30 days) to enable proactive maintenance and data protection strategies.

The link to Jupyter Notebook can be found here: [Link to Jupyter Notebook](/Project4.ipynb)

### Key Objectives
- Extract and process raw Backblaze quarterly ZIP files (Q1-Q4 2024)
- Perform exploratory data analysis (EDA) on drive records
- Engineer RUL classification targets at multiple prediction horizons
- Train and evaluate baseline machine learning models
- Provide actionable insights for maintenance planning

---

## About the Backblaze Dataset

### Data Source & Collection

**Backblaze Overview:**
Backblaze is a cloud backup and storage company that publishes quarterly hard drive failure statistics and raw data as part of their operational transparency initiative. The dataset represents real-world hard drive failure patterns from thousands of drives in active production data centers.

**Data Collection Method:**
- **Reporting Frequency:** Quarterly (Q1, Q2, Q3, Q4)
- **Data Format:** CSV files in quarterly ZIP archives
- **Monitoring Method:** Automatic monitoring of live production drives using SMART (Self-Monitoring, Analysis and Reporting Technology) metrics
- **Temporal Scope (this project):** Full calendar year 2024 (January 1 - December 31)

**Dataset Characteristics:**
| Metric | Value |
|--------|-------|
| Date Range | 2024-01-01 to 2024-12-31 |
| Unique Drive Models | 90+ (project focuses on top models) |
| Failure Rate (Overall) | ~0.5-1.0% (varies by quarter/model) |
| Drive Manufacturers | Seagate, Western Digital, Toshiba, Samsung, Hitachi |
| Drive Types | Enterprise-class HDDs (3.5" form factor) |
| Capacity Range | 4TB - 12TB+|

**Why Backblaze Data?**
- **Real-world production data** — Actual failure patterns from operating systems
- **Large scale** — Millions of drive observations reduce statistical noise
- **Publicly available** — Enables reproducible research and benchmarking
- **Temporal continuity** — Consistent monitoring methodology across years
- **Enterprise-relevant** — Failure patterns applicable to data center operations
- **Transparency** — Includes both successful and failed drives, enabling complete analysis

**Data Limitations:**
- **Survivorship bias** — Only captures drives that failed while in operation; doesn't include DOA (dead-on-arrival) drives
- **Selection bias** — Drives purchased by Backblaze may differ from general market
- **Missing causality** — Can't determine why specific failures occurred
- **Model variation** — Different models have vastly different failure rates
- **Environmental factors** — Temperature, humidity, usage patterns not captured in SMART data

---

## SMART Metrics Reference Guide

### Understanding SMART

**What is SMART?**
SMART stands for **Self-Monitoring, Analysis and Reporting Technology** — a monitoring system built into modern hard drives that tracks physical and operational attributes. These attributes provide early warning signs of drive degradation and imminent failure.

**SMART Levels:**
- **Raw Value** — Direct measurement from drive sensors (used in this project)
- **Normalized Value** — Vendor-adjusted value (0-100 scale; 100 = healthy)
- **Threshold** — Critical value set by manufacturer; exceeding triggers alarm
- **Status** — Current health assessment (OK, Caution, Critical)

### Selected SMART Metrics (13 Used in Project)

#### **1. SMART 5 — Reallocated Sectors Count**
- **Raw Value:** Number of sectors remapped to spare sectors
- **Why It Matters:** Indicates magnetic surface degradation and early failure risk
- **Failure Signal:** HIGH — Strong predictor of imminent failure
- **Typical Range:** 0-100 (threshold often: 100-200)
- **Interpretation:**
  - 0-10: Excellent (new/healthy drive)
  - 10-50: Normal (acceptable wear)
  - 50+: Concerning (elevated failure risk)

**Domain Knowledge:** Each time a read/write error occurs, the drive reallocates the damaged sector to a spare reserve. A continuously increasing count signals progressive surface degradation.

---

#### **2. SMART 9 — Power-On Hours**
- **Raw Value:** Total cumulative hours the drive has been powered on
- **Why It Matters:** Correlates with wear and tear; older drives fail more often
- **Failure Signal:** MODERATE — Weak direct predictor, but contextually important
- **Typical Range:** 0-100,000+ hours
- **Interpretation:**
  - 0-10,000 hours: New drive (< 1.5 years continuous operation)
  - 10,000-50,000 hours: Mid-life (1.5-5.7 years)
  - 50,000+ hours: Mature/aging (5.7+ years)

**Domain Knowledge:** Bathtub failure curve concept applies — drives fail more often when very new (manufacturing defects) or very old (wear-out). Peak reliability occurs in the middle lifespan.

---

#### **3. SMART 187 — Reported Uncorrectable Errors**
- **Raw Value:** Count of read/write errors that couldn't be corrected by error-correcting code (ECC)
- **Why It Matters:** **STRONGEST FAILURE PREDICTOR** in this dataset
- **Failure Signal:** CRITICAL — Nearly 100% predictive of imminent failure
- **Typical Range:** 0-10,000+ errors
- **Interpretation:**
  - 0: Perfect (no errors)
  - 1-10: Minor issues (may still recover)
  - 10+: Serious (failure imminent)

**Domain Knowledge:** When the drive's ECC subsystem fails to recover a sector, it indicates the magnetic layer is too degraded to read reliably. This is the "point of no return" for data integrity.

---

#### **4. SMART 188 — Command Timeout (Aborted Commands)**
- **Raw Value:** Number of commands that timed out due to drive not responding
- **Why It Matters:** Indicates electrical or firmware issues; drive not responding to host
- **Failure Signal:** HIGH — Strong indicator of imminent controller/firmware failure
- **Typical Range:** 0-1,000+ timeouts
- **Interpretation:**
  - 0: Excellent (drive responds instantly)
  - 1-10: Minor delays (occasional unresponsiveness)
  - 10+: Serious (frequent timeouts, drive flaking)

**Domain Knowledge:** Unlike surface errors (SMART 5/187), timeout errors suggest the drive's controller board or firmware is failing, not just magnetic degradation.

---

#### **5. SMART 189 — High Fly Writes**
- **Raw Value:** Count of times the read-write head came dangerously close to the disk surface
- **Why It Matters:** Indicates mechanical stress on the head arm; can precede head crash
- **Failure Signal:** MODERATE-HIGH — Predictor of mechanical failure
- **Typical Range:** 0-1,000+ events
- **Interpretation:**
  - 0-10: Good (stable head positioning)
  - 10-100: Concerning (intermittent vibration/misalignment)
  - 100+: Dangerous (frequent near-crashes, head crash imminent)

**Domain Knowledge:** Flying heights in modern HDDs are incredibly tight (~3-5 nanometers; human hair is ~75,000nm). Any vibration can cause head contact with platter, generating heat and data errors.

---

#### **6. SMART 196 — Reallocation Event Count**
- **Raw Value:** Number of times the drive detected a bad sector and reallocated it
- **Why It Matters:** Similar to SMART 5, but counts *events* rather than *cumulative sectors*
- **Failure Signal:** HIGH — Indicator of progressive surface degradation
- **Typical Range:** 0-200+ events
- **Interpretation:**
  - 0-5: Excellent
  - 5-20: Normal wear
  - 20+: Elevated risk

**Domain Knowledge:** Different vendors implement SMART 5 and 196 differently. Some count only reallocated sectors; others count reallocation events. Together they paint a complete picture of drive surface health.

---

#### **7. SMART 197 — Current Pending Sector Count**
- **Raw Value:** Number of sectors with read errors that haven't yet been reallocated
- **Why It Matters:** Represents sectors in limbo — tried to fix but not yet permanently bad
- **Failure Signal:** CRITICAL — Immediate action required; failure very soon
- **Typical Range:** 0-10,000+ pending sectors
- **Interpretation:**
  - 0: Perfect (all errors resolved or none detected)
  - 1-10: Watch closely (errors not yet critical)
  - 10+: Backup immediately (relocation failing; failure imminent)

**Domain Knowledge:** When ECC can still correct an error but the sector keeps failing on rereads, it goes to "pending" status. If reallocation succeeds, it moves to reallocated count. If it fails repeatedly, it signals loss of drive control.

---

#### **8. SMART 198 — Offline Uncorrectable Sector Count**
- **Raw Value:** Number of sectors that can't be read in offline diagnostic mode
- **Why It Matters:** Sectors completely inaccessible; data permanently lost in these areas
- **Failure Signal:** CRITICAL — Drive is failing/failed
- **Typical Range:** 0-10,000+ sectors
- **Interpretation:**
  - 0: Healthy (no offline errors)
  - 1+: Serious (data loss occurring; backup critical)
  - 100+: Drive unusable (retirement recommended)

**Domain Knowledge:** Unlike online errors (SMART 187), offline errors occur during self-tests. Any positive value means parts of the drive platter are physically damaged or magnetically degraded beyond recovery.

---
## Methodology

### 1. Data Extraction & Loading

**Source Data:**
- Backblaze quarterly CSV files in ZIP archives (data_Q1_2024.zip through data_Q4_2024.zip)
- ~3.4M records with 13 selected SMART metrics and drive metadata

**Multi-Encoding File Handling:**
- Implemented robust multi-encoding fallback strategy (UTF-8 → Latin-1 → ISO-8859-1 → CP1252)
- Prevents `UnicodeDecodeError` when reading CSV files with mixed encodings
- Uses BytesIO for reliable in-memory file operations

**Feature Selection (13 Columns):**
```python
[
    'date',              # Observation date
    'serial_number',     # Unique drive identifier
    'model',             # Drive model (e.g., ST4000DM000)
    'capacity_bytes',    # Drive capacity
    'failure',           # Binary failure indicator (0/1)
    'smart_5_raw',       # Reallocated sectors count
    'smart_9_raw',       # Power-on hours
    'smart_187_raw',     # Reported uncorrectable errors
    'smart_188_raw',     # Command timeout count
    'smart_189_raw',     # High fly writes
    'smart_196_raw',     # Reallocation event count
    'smart_197_raw',     # Current pending sector count
    'smart_198_raw'      # Offline uncorrectable sector count
]
```

**Temporal Tracking:**
- Added `data_quarter` column during extraction to track which ZIP each record originated from
- Enables temporal validation and prevents data leakage

### 2. Data Quality Assessment

**Duplicate Detection & Removal:**
- Identified and removed duplicate records
- Preserved data integrity through consistent handling

**Missing Value Analysis:**
- Analyzed missing data percentages across all features
- Applied median imputation for SMART columns
- Dropped columns with >95% missing values (none identified)

**Outlier Detection:**
- Used Interquartile Range (IQR) method: `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`
- Retained outliers for analysis (may indicate drive degradation)

**Data Type Coercion:**
- Converted date strings to datetime64 for temporal analysis
- Coerced SMART and capacity columns to numeric types
- Applied error handling for malformed values

### 3. Exploratory Data Analysis (EDA)

**Temporal Analysis:**
- Analyzed failure distribution across 365 days of 2024
- Computed daily failure rates and 7-day rolling averages
- Identified quarterly trends in failure patterns

**Univariate Analysis:**
- Examined distributions of all 13 SMART features
- Computed descriptive statistics (mean, median, std, min, max)
- Identified skewed distributions (e.g., power-on hours)

**Class Balance Assessment:**
- Observed severe class imbalance (0.0439% failures at 2-day horizon)
- Imbalance ratio: ~990:1 (non-failures to failures)
- Expected in operational datasets; impacts model evaluation metrics

**Feature-Failure Correlation:**
- Computed Pearson correlations between SMART metrics and failure
- Performed Mann-Whitney U tests to identify significant feature differences
- Identified smart_187_raw (uncorrectable errors) as most predictive feature

**Correlation Analysis:**
- Built correlation matrix across all numeric features
- Identified redundant features and feature interactions
- Detected moderate collinearity patterns

### 4. RUL Bucketing & Target Engineering

**Multi-Horizon Classification Strategy:**
Instead of binary failure prediction, implemented 4-horizon RUL classification to provide actionable time windows:

```
horizon = 2 days:   Are failures predicted to occur within 2 days? (urgent)
horizon = 7 days:   Are failures predicted to occur within 7 days? (soon)
horizon = 14 days:  Are failures predicted to occur within 14 days? (monitor)
horizon = 30 days:  Are failures predicted to occur within 30 days? (plan)
```

**Target Construction:**
1. For each drive (serial_number), identify first observed failure date
2. For each record, compute days_to_failure = failure_date - record_date
3. Remove post-failure records (days_to_failure < 0)
4. Create 4 binary targets: `fail_within_{2,7,14,30}d = (days_to_failure ≤ horizon)`

**Positive Class Distribution:**
- 2-day horizon:   1,677 failures (0.0439%)
- 7-day horizon:   4,377 failures (0.1147%)
- 14-day horizon:  8,289 failures (0.2172%)
- 30-day horizon: 17,248 failures (0.4520%)

### 5. Train-Test Split Strategy

**Temporal/ZIP-Based Split (Production-Realistic):**
- **Training Set:** Q1 + Q2 + Q3 2024
- **Test Set:** Q4 2024
- **Benefit:** Validates model generalization to future time periods
- **No Data Leakage:** Q4 test dates are strictly after Q1-Q3 training dates

**Alternative Available:** Random 80/20 split (commented out, not used)

### 6. Baseline Model Development

**Model Architecture:**
- **Algorithm:** Logistic Regression (simple, interpretable baseline)
- **Scaling:** StandardScaler (fit on train, transform train/test)
- **Imputation:** Median imputation for missing feature values
- **Regularization:** Default L2 penalty (C=1.0)
- **Hyperparameters:** max_iter=1000, random_state=42, n_jobs=-1 (parallel)

**Training Process:**
1. For each horizon (2, 7, 14, 30 days):
   - Extract target: `y = fail_within_{horizon}d`
   - Split features: X_train (Q1-Q3), X_test (Q4)
   - Scale training and test features
   - Train Logistic Regression
   - Store model, test data, and metrics

**Evaluation Metrics:**
- **Accuracy:** Overall correctness (less relevant for imbalanced data)
- **ROC-AUC:** Threshold-independent discrimination ability
- **F1-Score:** Harmonic mean of precision and recall (accounts for imbalance)
- **Confusion Matrix:** Breakdown of TP, TN, FP, FN
- **Classification Report:** Per-class precision, recall, F1-score

---

### Model Performance Summary

| Horizon (days) | Samples | Positives | Positive Rate (%) | Accuracy | ROC-AUC | F1-Score | Status |
|:--------------:|:-------:|:---------:|:-----------------:|:--------:|:-------:|:--------:|:------:|
| 2 | 3,816,204 | 1,677 | 0.0439% | 0.9995 | 0.8664 | 0.0000 | ok |
| 7 | 3,816,204 | 4,377 | 0.1147% | 0.9989 | 0.8576 | 0.0000 | ok |
| 14 | 3,816,204 | 8,289 | 0.2172% | 0.9980 | 0.8359 | 0.0011 | ok |
| 30 | 3,816,204 | 17,248 | 0.4520% | 0.9957 | 0.8104 | 0.0261 | ok |
  
### Notebook Organization

**backblaze_eda_exploratory.ipynb (Main Notebook)**
- **Cell 1:** Dependencies & Setup
- **Cells 2A-2D:** Data Extraction from ZIPs with temporal tracking
- **Cell 2C:** Initial EDA
- **Cell 3:** Data quality assessment & cleaning
- **Cell 4:** Data Distribution analysis
- **Cell 5:** Class balance analysis
- **Cell 6:** Temporal failure trends
- **Cell 7:** Correlation analysis
- **Cell 8:** Feature engineering demonstrations
- **Cell 9:** Failure Relationships
- **Cell 10:** Baseline model development (4 horizons, Logistic Regression)
- **Cell 11:** Model evaluation (metrics, ROC curves, coefficients)
- **Cell 13:** Executive summary & recommendations

---

## Usage Instructions

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy ipython jupyter
```

### Running the Notebook

1. **Ensure ZIP files are in working directory:**
   ```bash
   ls data_Q*.zip  # Should list 4 files
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook backblaze_eda_exploratory.ipynb
   ```

3. **Execute cells sequentially:**
   - Cells 1-7: Data loading, extraction, EDA (no hyperparameter tuning)
   - Cells 8-13: Feature engineering, modeling, evaluation

4. **Output:**
   - EDA visualizations (distributions, correlations, time series)
   - ROC curves for all 4 horizons
   - Cleaned training CSV in `rul_output/` directory
   - Model metrics and recommendations printed to console

### Customization

**Change Drive Model:**
```python
# In Cell 2D, modify:
user_selected_models = ['ST4000DM000']  # Single model
# OR
user_selected_models = ['ST4000DM000', 'ST12000NM0007']  # Multiple models
# OR set to None to use top N failing models
```

**Adjust Temporal Split:**
The code supports both temporal and random splits. To use random split, uncomment lines in Cell 10.

---

## Key Insights & Recommendations

### High Priority Actions
1. **Use multi-horizon approach** — Different prediction windows enable tailored maintenance strategies
2. **Focus on 2-day and 30-day horizons** — Provides both urgent and preventive windows
3. **Monitor ROC-AUC primarily** — Ignore high accuracy; focus on discrimination ability
4. **Apply class weighting per horizon** — Improve minority recall without sacrificing specificity

### Future Work
1. Extend with advanced algorithms (Random Forest, XGBoost, LightGBM)

---
## References

- **Backblaze Blog:** https://www.backblaze.com/blog/
- **SMART Attributes:** https://en.wikipedia.org/wiki/S.M.A.R.T.
- **RUL Prediction:** https://en.wikipedia.org/wiki/Remaining_useful_life
- **Class Imbalance:** https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

---
