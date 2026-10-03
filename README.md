# Healthcare Fraud Detection (Milestone 1)

## 📌 Project Overview
This repository contains **Milestone 1** of the Healthcare Fraud Detection project for the Data and Infrastructure (2026) course. The primary goal of the overall project is to build an end-to-end data engineering & ML pipeline capable of identifying fraudulent healthcare claims, minimizing financial leakage, and isolating anomalous billing behavior. We will be working with scikit-learn classifiers and will initially trial an XGBoost algorithm. Future work may include comparing this to a Random Forest approach. Tabular data (like healthcare fraud records) is famously dominated by tree-based ensembles, so deep learning would likely add a lot of complexity for very little, if any, performance gain.

* **Notebook Location:** [`notebooks/healthcare_fraud_milestone_1.ipynb`](notebooks/healthcare_fraud_milestone_1.ipynb)

---

## 🏗️ Data Architecture & Infrastructure

### 1. Raw Data Storage (Criterion 1)
* **Storage System/Location:** Raw data is stored within an immutable **Google Cloud Storage (GCS)** bucket architecture.
* **Justification:** GCS serves as our central data lake. Object storage is highly appropriate for this stage because it provides secure, cost-effective, high-durability storage for raw, unmodified healthcare transactional records before downstream transformation.

### 2. Processed Data Storage & File Formats (Criterion 2 & 8)
* **Storage Location:** Processed files are hosted in segregated data directories within our GCS environment.
* **File Formats:** 
  * **CSV Format (`healthcare_fraud_splits.csv`):** Maintained for standard portability and universal downstream compatibility.
  * **Parquet Format (`healthcare_fraud_features.parquet`):** Implemented for forward utility and production pipelines.
* **Justification:** Storing processed data as **Parquet** introduces a columnar schema layout that optimizes query execution, preserves exact data types, and reduces storage overhead through efficient internal compression (`pyarrow`).

### 3. Database vs. Object Storage Decision (Criterion 3)
* **Chosen Solution:** **Object Storage (Google Cloud Storage)** combined with structured flat files (`.csv` and `.parquet`).
* **Justification:** Because our primary workload patterns involve batch machine learning training rather than frequent, concurrent transactional point-lookups (OLTP), managed object storage eliminates unnecessary database computing costs and operational overhead while scaling seamlessly to larger data volumes.

### 4. Data Versioning Strategy (Criterion 4)
* **Mechanism:** Dual-layer data tracking leveraging **native GCS Bucket Versioning** paired with an **explicit folder path naming structure** natively implemented in our notebook.
* **Tracking Rules:** 
  * **Explicit Folder Isolation:** Datasets are separated into distinct directory targets using a deterministic date identifier runtime variable (`PROCESSING_DATE = date.today().isoformat()`). Files are uploaded cleanly to targets like `gs://[bucket-name]/YYYY-MM-DD/splits/` and `gs://[bucket-name]/YYYY-MM-DD/features/`. This provides immediate visual lineage tracking for each pipeline run.
  * **Object-Level Versioning:** Object versioning is active on our GCS bucket. When files are overwritten or updated within a specific date directory, GCS automatically retains a historical log and stamps older files with a unique `generation` ID, protecting the pipeline against silent data mutations or accidental deletion.

### 5. Data Access & Credentials (Criterion 5)
* **Access Interface:** The application accesses cloud assets programmatically using custom Python wrappers (`upload_to_gcs`) and the official Google Cloud Python SDK client library (`google-cloud-storage`).
* **Authentication Mechanisms:** 
  * **Kaggle API:** Authenticated via runtime generation or passing of a secure `kaggle.json` credential file containing API keys.
  * **GCP/GCS Access:** Authenticated interactively within the notebook using the runtime credential provider framework via `google.colab.auth.authenticate_user()`. This verifies organizational IAM permissions without hardcoding secrets.

---

## 🧪 Data Engineering & Methodology

### 6. Data Split & Validation Strategy (Criterion 6)
* **Splitting Protocol:** The data is clean-split into explicit **Train (80%), Dev/Validation (10%), and Test (10%)** subsets using `sklearn.model_selection.train_test_split`.
* **Leakage Prevention:** 
  * Splitting is executed via a `random_state=42` and is strictly **stratified** by our target column (`stratify=df_health_fraud["Is_Fraud"]`) to preserve class proportions across folds.
  * To ensure **zero lookahead or data leakage**, all engineered historical velocity statistics are computed *only* on the training dataset (`train_df`). These aggregated metrics are then mapped to the validation and test datasets strictly as lookups.

  ## Validation Strategy

To evaluate the generalization performance of our tree-based models (XGBoost and Random Forest) and prevent data leakage, an **8-fold Stratified Cross-Validation** approach was implemented. 

### Architectural Decisions & Efficiency
* **Dynamic In-Loop Validation:** Rather than pre-saving multiple slices of the dataset into a dictionary or explicit structures beforehand, the cross-validation logic processes splits dynamically within the training loop.
* **Memory Efficiency:** XGBoost and Random Forests can be memory-heavy when building deep trees. Pre-saving multiple copies of your dataset into a dictionary eats up RAM unnecessarily; this loop-based approach processes and discards data fold-by-fold.
* **Pipelines for Data Safety:** By passing data slices straight into a scikit-learn `Pipeline`, data transformations are isolated to individual folds, guaranteeing zero data leakage without requiring manual `.copy()` calls on the underlying data frames. 

### 7. Feature Descriptions (Criterion 7)

Below are the base columns and engineered features mapped within our machine learning architecture:

| Feature Name | Data Type | Feature Type | Description & Engineering Origin |
| :--- | :--- | :--- | :--- |
| **Is_Fraud** | Binary (`int64`) | Target Variable | 1 indicates a confirmed fraudulent claim; 0 indicates a legitimate claim. Enforced as a strict integer indicator. |
| **Provider_ID** | Categorical (`object`) | Base Feature | Unique identifier hash for the healthcare facility or physician. Used as the grouping key for behavioral velocity mapping. |
| **Claim_Submission_Date** | Temporal (`object`) | Base Feature | The raw timestamp when the healthcare claim was filed. Completely dropped from the final feature matrix to prevent leakage. |
| **Submission_Month** | Numerical (`int64`) | Engineered | Extracted month component from `Claim_Submission_Date` to safely capture seasonality without lookahead bias. |
| **Is_Submission_Weekend** | Binary (`int64`) | Engineered | Flag (0/1) identifying if the claim was submitted on a Saturday or Sunday to uncover automated off-hours submission anomalies. |
| **Diagnosis_Code** | Categorical (`object`) | Base Feature | High-cardinality standard medical billing code. Managed upstream via Smooth Target Encoding to prevent dimensional explosion. |
| **Procedure_Code** | Categorical (`object`) | Base Feature | High-cardinality standard operational treatment code. Managed upstream via Smooth Target Encoding. |
| **Provider_Specialty** | Categorical (`object`) | Base Feature | High-cardinality domain specialty of the provider. Missing values imputed as 'Unknown' to capture operational anomalies; managed via Target Encoding. |
| **Patient_State** | Categorical (`object`) | Base Feature | Geographic state indicator of the patient. Managed via Smooth Target Encoding to extract regional risk baseline frequencies. |
| **Insurance_Type** | Categorical (`object`) | Base Feature | Low-cardinality classification of the policy. Missing values imputed as 'Unknown'; processed downstream via One-Hot Encoding. |
| **Visit_Type** | Categorical (`object`) | Base Feature | Low-cardinality classification of the medical encounter. Processed downstream via One-Hot Encoding (`drop='first'`). |
| **Claim_Amount** | Numerical (`float64`) | Base Feature | The continuous monetary value requested by the provider. Cast explicitly to float64 to preserve precision variance. |
| **Approved_Amount** | Numerical (`float64`) | Base Feature | The continuous monetary value approved for payout. Cast explicitly to float64 to preserve precision variance. |
| **Length_of_Stay** | Numerical (`float64`) | Base Feature | Total continuous duration of the hospital or clinic encounter. Cast explicitly to float64. |
| **Number_of_Claims_Per_Provider_Monthly** | Numerical (`int64`) | Base Feature | Operational load count showing billing volume per provider within a standard monthly window. |
| **Chronic_Condition_Flag** | Numerical (`int64`) | Base Feature | Binary numeric flag indicating the presence of pre-existing patient underlying health risks. |
| **Prior_Visits_12m** | Numerical (`float64`) | Base Feature | The number of medical encounters recorded for the patient in the prior year. Missing values safely imputed to 0. |
| **Hist_Pct_Fast_Claims** | Numerical (`float64`) | Engineered | Provider Velocity Profile: Proportion of a provider's historical claims submitted in under 5 days, calculated strictly on train folds. |
| **Hist_Mean_Lag** | Numerical (`float64`) | Engineered | Provider Velocity Profile: A provider's historical average claim submission turnaround window, mapped via leak-free lookups. |
| **Hist_Lag_Std** | Numerical (`float64`) | Engineered | Provider Velocity Profile: The historical standard deviation of a provider's turnaround times to capture behavioral inconsistency. |
| **Current_vs_Hist_Mean_Diff** | Numerical (`float64`) | Engineered | Behavioral Tally: The mathematical delta between the current claim's submission speed and the provider's historical mean lag window. |
| **Current_Speed_Z_Score** | Numerical (`float64`) | Engineered | Behavioral Anomaly Weight: The statistical Z-Score measuring how many standard deviations the current transaction speed deviates from the provider's normal workflow routine. |



### 💡 Engineering Rationale: Why We Engineered Provider Velocity Profiles - Feature Selection & Engineering Report: The Timeline Trap

## 📊 Exploratory Data Analysis & Feature Profiling
Analysis of `Days_Between_Service_and_Claim` showed differences between legitimate and fraudulent claim timelines. A leak-free **Historical Provider Velocity Profile** framework is implemented using strict partition separation, train-only profiling, and lookup mapping. 

During initial feature profiling, an evaluation of the temporal feature `Days_Between_Service_and_Claim` revealed a stark, anomalous separation between legitimate and fraudulent transactions:

### Profiling: Days_Between_Service_and_Claim
* **Legitimate Claims (`Is_Fraud = 0`):** Mean lag of **15.45 days** (Median: 15.0). Range spans from 2 to 29 days. Zero values: 0.
* **Fraudulent Claims (`Is_Fraud = 1`):** Mean lag of **2.97 days** (Median: 3.0). Range strictly capped between 0 and 6 days. Zero values: 115.

---

## 🪤 The Operational "Timeline Trap" & Target Leakage

### 1. The Training Shortcut
If the raw `Days_Between_Service_and_Claim` feature is passed directly into a machine learning model, the algorithm finds an artificial structural shortcut. Because **100% of historical fraud cases are clustered under 6 days**, the classifier achieves a near-perfect evaluation score (e.g., ROC-AUC > 0.99) by ignoring medical data entirely and routing decisions solely through this single time metric.

### 2. Production Failure: Who Gets Penalized?
Deploying this raw feature creates a catastrophic disconnect in a live environment:
* **Penalizing the Efficient Physician:** Fraud operations move aggressively to cash out before detection systems trigger. However, speed itself is not unique to fraud. **An outstanding, prompt physician who maintains pristine, real-time administrative workflows and submits claims within 48 hours of a patient visit will be falsely flagged as a fraudster.**
* **The Operational Blindspot:** The model never actually learns the underlying medical, financial, or geographic hallmarks of fraud; it merely penalizes operational efficiency.

---

## 🛠️ Infrastructure Solution: Provider Velocity Profiling

To resolve this contradiction and protect legitimate, high-performing physicians, the raw, transaction-level metric was **dropped entirely** from the feature matrix and replaced with a leak-free **Historical Provider Velocity Profile** framework.

### How the Leak-Free Pipeline Works (Milestone 1 Implementation)
Instead of scoring a claim based on its *current* submission speed, the pipeline aggregates a provider's historical behavioral footprint:

1. **Strict Partition Separation:** The dataset is split into Train (80%), Dev (10%), and Test (10%).
2. **Train-Only Profiling:** Historical aggregates are calculated **strictly** using data inside the training split to prevent lookahead contamination:
   * `Hist_Pct_Fast_Claims`: The proportion of a provider's past claims submitted in under 5 days.
   * `Hist_Mean_Lag`: A provider's historical average submission turnaround.
3. **Downstream Lookup Mapping:** These metrics are mapped to the validation and testing partitions as fixed behavioral characteristics. If a provider is unseen in the training window, they receive safe baseline indicators (`global_mean_lag`).

### Core Benefit
By transitioning from an instance-level shortcut to a provider-level behavioral profile, the model is forced to evaluate actual clinical anomalies, geographic patterns, and financial structures—ensuring stable, secure scaling in a live production environment without penalizing prompt healthcare providers.

### 🎛️ Dynamic Transaction Tallying (Defeating the Blind Spot)
While pure provider reputation profiles eliminate target leakage, completely ignoring the current claim's speed creates an operational blind spot where the model evaluates a provider's history but ignores current transaction anomalies. 

To bridge this gap without reintroducing the "Timeline Trap," our pipeline extracts leak-free relational velocity metrics (`Current_vs_Hist_Mean_Diff` and `Current_Speed_Z_Score`). Instead of analyzing raw transaction speed, the system evaluates how far the current claim's submission timeline deviates from that specific provider's established historical standard deviation. This allows our infrastructure to instantly flag a typically meticulous provider who suddenly exhibits high-velocity billing bursts, forcing the algorithm to balance historical baseline reputational data with live operational anomalies safely.


### 8. Data Types and Serialization Formats (Criterion 8)
To guarantee optimal execution efficiency, schema validation, and storage portability across our cloud environment, data types are strictly cast and managed.

#### Raw Pandas Data Types Mapping
The raw source includes standard pandas data types for identifiers, numerical counts, amounts, and flags ranging from `object`, `int64`, and `float64`.

#### Pipeline Type Specifications
* **Categorical Dimensions (`Provider_ID`)**: Handled computationally as high-cardinality structural strings, managed upstream of modeling via smooth Target Encoding configurations to avoid dimensional scaling explosion.
* **Numerical Metrics** (`Claim_Amount`, `Approved_Amount`, `Hist_Mean_Lag`, `Current_vs_Hist_Mean_Diff`, `Current_Speed_Z_Score`, etc.): Represented and cast exclusively using 64-bit continuous floating-point descriptors (`float64`) or integers (`int64`) to preserve precision variance during transformations.
* **Binary Markers (`Is_Fraud`, `Is_Submission_Weekend`)**: Structuralised natively as clean binary indicators (`int` 0/1) for zero-entropy processing.

#### Serialization Formats
* **Interchange Format (`.csv`)**: The partitioned matrix splits are structured into `healthcare_fraud_splits.csv` to ensure cross-platform human-readable portability.
* **Production/Storage Format (`.parquet`)**: The final features are serialized into `healthcare_fraud_features.parquet`. Using Apache Parquet ensures strict metadata type preservation, column-oriented disk storage layouts for fast batch training I/O, and efficient Snappy compression footprints within our GCS data lake layers.

---

### 9. Reproducibility of Data Collection (Criterion 9)
* **Data Source:** Programmatically pulled from the official **Kaggle API**.
* **Collection Steps:** 
  1. Initialize connection to Kaggle via the execution environment using automated API credentials.
  2. Download the compressed raw archive directly into the local Colab runtime space.
  3. Extract files and stage them to the primary raw Google Cloud Storage repository path.

### 10. Reproducibility of Preprocessing & Pipeline Steps (Criterion 10)
Preprocessing executes a strict sequential pipeline including datetime parsing, partition isolation, profile generation, imputation, and cloud serialization. Distributional skews and imbalances are managed via log-transformations, stratified splits, and target encoding.

To fully recreate our clean feature matrices from the raw source files, the preprocessing execution block in our notebook runs a strict sequential pipeline:
1. **Datetime Parsing:** Converts `Claim_Submission_Date` into a standard pandas datetime format to engineer `Submission_Month` and `Is_Submission_Weekend`.
2. **Strict Partition Isolation:** Splits the source matrix into Train, Dev, and Test dataframes using `sklearn.model_selection.train_test_split`.
3. **Safe Profile Generation:** Groups the training set (`train_df`) by `Provider_ID` to generate historical velocity statistics (`Hist_Pct_Fast_Claims`, `Hist_Mean_Lag`, `Hist_Lag_Std`).
4. **Imputation & Fallback Application:** Merges the profiles back into all three splits. Any provider completely unseen during the training sequence is imputed with safe global metrics (`global_pct_fast`, `global_mean_lag`, `global_std_lag`) derived strictly from the training collection.
5. **Cloud Serialization:** Combined tracking frames are tagged with their split identity and saved locally before uploading to GCS as paired `.csv` and optimized `.parquet` targets under the active `PROCESSING_DATE` directory namespace.

### 📈 Distributional Skew Management Strategy

Initial profiling of the feature matrix reveals severe distributional imbalances. The engineering pipeline is structured to explicitly handle these specific variations:

* **Right-Skewed Financial Targets (`Claim_Amount`, `Approved_Amount`):** 
  * **The Issue:** Legitimate claims cluster heavily around low values (Median: \$385.00), whereas fraudulent operations display an aggressive right-tail distribution with extreme maximum outliers scaling up to \$6,590.70.
  * **The Mitigation:** These unbounded financial metrics are passed through monotonic log-transformations (or robust scalers) within our training pipeline to compress variance and stabilize gradient updates.
* **Highly Imbalanced Target Slices (`Is_Fraud`):**
  * **The Issue:** Severe class imbalance exists with 9,171 normal instances to only 829 fraud anomalies (~8.3% base fraud rate).
  * **The Mitigation:** The dataset partition pipeline forces strict stratified mapping splits and dynamically computes an empirical training weight offset (`scale_pos_weight = num_neg / num_pos`) passed directly into the tree-hist algorithm to prevent structural convergence bias.
* **High-Cardinality Sparsity (`Provider_ID`, `Diagnosis_Code`, `Procedure_Code`):**
  * **The Issue:** Tracking individual categorical dimensions introduces massive high-cardinality dimensionality explosion if passed to naive one-hot encoding matrices. Further the order or cardinality carry no inherent informational value and can throw a model off balance if it associates the numerical ordinal value with informational importance or relevance. 
  * **The Mitigation:** The pipeline applies a regularized `TargetEncoder(smooth=10.0)` setup, tracking specific historical category conditional targets rather than expanding sparse structural columns.

---

## 🚀 Getting Started & Execution

Because this project relies on **Google Colab** wrappers and authentication protocols, executing it directly inside Colab is the recommended path.

### Execution via Google Colab (Recommended)

1. **Launch the Workspace:**
   Upload the notebook in google colab and fill in the details for your project and bucket name in GCS. You will need a Kaggle api key to download the dataset and google authentication to connect to GCS.

   If running this lab in Jupyter notebook additional configuration may be required which is not covered in this example.

2. **Kaggle Authentication:**
   * When executing the data collection cell, ensure you upload or provide your `kaggle.json` API token file to allow programmatical data downloading.

3. **Google Cloud Platform (GCP) Authentication:**
   * Run the interactive cell containing:
     ```python
     from google.colab import auth
     auth.authenticate_user()
     ```
   * Follow the browser sign-in prompt using your active cloud project credentials to allow the runtime kernel to write/read from the target GCS bucket.

