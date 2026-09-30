# Healthcare Fraud Detection (Milestone 1)

## 📌 Project Overview
This repository contains **Milestone 1** of the Healthcare Fraud Detection project for the Data and Infrastructure (2026) course. The primary goal of this project is to build an end-to-end data engineering pipeline capable of identifying fraudulent healthcare claims, minimizing financial leakage, and isolating anomalous billing behavior.

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

### 7. Feature Descriptions (Criterion 7 & 8)
Below are some of the base columns and engineered features mapped within our machine learning architecture:

| Feature Name | Data Type | Feature Type | Description & Engineering Origin |
| :--- | :--- | :--- | :--- |
| `Provider_ID` | Categorical | Base Feature | Unique identifier hash for the healthcare facility or physician. |
| `Is_Fraud` | Binary | **Target Variable** | 1 indicates a confirmed fraudulent claim; 0 indicates a legitimate claim. |
| `Claim_Submission_Date` | Temporal | Base Feature | The raw timestamp when the healthcare claim was filed. |
| `Submission_Month` | Numerical | Engineered | Extracted month component from `Claim_Submission_Date` to capture seasonality. |
| `Is_Submission_Weekend` | Binary | Engineered | Flag (0/1) identifying if the claim was submitted on a Saturday or Sunday. |
| `Hist_Pct_Fast_Claims` | Numerical | Engineered | Percentage of claims processed under 5 days (`Days_Between_Service_and_Claim`).

### 💡 Engineering Rationale: Why We Engineered Provider Velocity Profiles

In healthcare fraud detection, analyzing isolated, individual claims rarely reveals fraudulent patterns. True fraud signals usually emerge from **behavioral anomalies over time at the provider level**. We engineered the historical velocity profiles for the following business and data design reasons:

* **Capturing Behavioral Velocity:** Legitimate providers typically follow steady, predictable administrative rhythms. Fraudulent rings often engage in "burst" billing—submitting huge volumes of claims immediately following a supposed patient service to cash out before detection systems trigger an audit.
* **Isolating Operational Risk Flags:** By computing `Hist_Pct_Fast_Claims` (claims filed under 5 days) alongside `Hist_Mean_Lag`, we can mathematically highlight providers who exhibit statistical anomalies in their billing velocity compared to industry standards.
* **Strict Leakage Minimization:** Calculating these metrics *solely* within the training partition (`train_df`) and mapping them downstream via safe global fallbacks (`global_pct_fast`, etc.) guarantees that our model cannot "peek" into the evaluation windows. This simulates a realistic production deployment where future provider trends remain entirely unseen.
* **Feature engineering decision:** An exploratory data analysis of the Days_Between_Service_and_Claim feature revealed a distinct separation in submission lag between legitimate claims (averaging 15.45 days) and fraudulent claims (averaging 2.97 days). Including this raw feature in predictive models introduces target leakage and the "Timeline Trap," where live production systems cannot calculate absolute lag metrics in real time. Replacing the raw metric with a leak-free provider velocity profiling framework prevents data shortcuts and ensures secure scaling in production.

### 8. Reproducibility of Data Collection (Criterion 9)
* **Data Source:** Programmatically pulled from the official **Kaggle API**.
* **Collection Steps:** 
  1. Initialize connection to Kaggle via the execution environment using automated API credentials.
  2. Download the compressed raw archive directly into the local Colab runtime space.
  3. Extract files and stage them to the primary raw Google Cloud Storage repository path.

### 9. Reproducibility of Preprocessing & Pipeline Steps (Criterion 10)
To fully recreate our clean feature matrices from the raw source files, the preprocessing execution block in our notebook runs a strict sequential pipeline:
1. **Datetime Parsing:** Converts `Claim_Submission_Date` into a standard pandas datetime format to engineer `Submission_Month` and `Is_Submission_Weekend`.
2. **Strict Partition Isolation:** Splits the source matrix into Train, Dev, and Test dataframes using `sklearn.model_selection.train_test_split`.
3. **Safe Profile Generation:** Groups the training set (`train_df`) by `Provider_ID` to generate historical velocity statistics (`Hist_Pct_Fast_Claims`, `Hist_Mean_Lag`, `Hist_Lag_Std`).
4. **Imputation & Fallback Application:** Merges the profiles back into all three splits. Any provider completely unseen during the training sequence is imputed with safe global metrics (`global_pct_fast`, `global_mean_lag`, `global_std_lag`) derived strictly from the training collection.
5. **Cloud Serialization:** Combined tracking frames are tagged with their split identity and saved locally before uploading to GCS as paired `.csv` and optimized `.parquet` targets under the active `PROCESSING_DATE` directory namespace.

---

## 🚀 Getting Started & Execution

Because this project relies on **Google Colab** wrappers and authentication protocols, executing it directly inside Colab is the recommended path to prevent environment fragmentation.

### Execution via Google Colab (Recommended)

1. **Launch the Workspace:**
   Click the button below to launch the milestone notebook directly inside your browser:
   [![Open In Colab](https://google.com)](https://google.com)

2. **Kaggle Authentication:**
   * When executing the data collection cell, ensure you upload or provide your `kaggle.json` API token file to allow programmatical data downloading.

3. **Google Cloud Platform (GCP) Authentication:**
   * Run the interactive cell containing:
     ```python
     from google.colab import auth
     auth.authenticate_user()
     ```
   * Follow the browser sign-in prompt using your active cloud project credentials to allow the runtime kernel to write/read from the target GCS bucket.

