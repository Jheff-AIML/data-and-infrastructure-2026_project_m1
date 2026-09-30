# Healthcare Fraud Detection (Milestone 1)

## 📌 Project Overview
This repository contains **Milestone 1** of the Healthcare Fraud Detection project for the Data and Infrastructure (2026) course. The primary goal of the overall project is to build an end-to-end machine learning pipeline capable of identifying fraudulent healthcare claims, minimizing financial leakage, and isolating anomalous billing behavior.

* **Notebook Location:** [`notebooks/healthcare_fraud_milestone_1.ipynb`](notebooks/healthcare_fraud_milestone_1.ipynb)

---

## 🏗️ Data Architecture & Infrastructure

### 1. Raw Data Storage (Criterion 1)
* **Storage System/Location:** Raw data is stored within an immutable **Google Cloud Storage (GCS)** bucket architecture.
* **Justification:** GCS serves as our central data lake. Object storage is highly appropriate for this stage because it provides secure, cost-effective, high-durability storage for raw, unmodified healthcare transactional records before downstream transformation.

### 2. Processed Data Storage & File Formats (Criterion 2 & 8)
* **Storage Location:** Processed files are hosted in segregated data directories within our GCS environment.
* **File Formats:** 
  * **CSV Format:** Maintained for standard portability and universal downstream compatibility.
  * **Parquet Format:** Implemented for forward utility and production pipelines.
* **Justification:** Storing processed data as **Parquet** introduces a columnar schema layout that optimizes query execution, preserves exact data types (integers, floats, categories), and reduces storage overhead through efficient internal compression (Snappy/Gzip).

### 3. Database vs. Object Storage Decision (Criterion 3)
* **Chosen Solution:** **Object Storage (Google Cloud Storage)** combined with structured flat files (`.csv` and `.parquet`).
* **Justification:** Because our primary workload patterns involve batch machine learning training rather than frequent, concurrent transactional point-lookups (OLTP), managed object storage eliminates unnecessary database computing costs and operational overhead while scaling seamlessly to larger data volumes.

### 4. Data Versioning Strategy (Criterion 4)
* **Mechanism:** Dual-layer data tracking leveraging **native GCS Bucket Versioning** paired with an **explicit folder path naming structure** natively implemented in our notebook.
* **Tracking Rules:** 
  * **Explicit Folder Isolation:** Datasets are separated into distinct directory targets using a deterministic date identifier (e.g., `gs://[bucket-name]/data_YYYYMMDD/`). This provides immediate visual lineage tracking for each pipeline run.
  * **Object-Level Versioning:** Object versioning is active on our GCS bucket. When files are overwritten or updated within a specific date directory, GCS automatically retains a historical log and stamps older files with a unique `generation` ID, protecting the pipeline against silent data mutations or accidental deletion.

### 5. Data Access & Credentials (Criterion 5)
* **Access Interface:** The application accesses cloud assets programmatically using the official Google Cloud Python SDK client library (`google-cloud-storage`).
* **Authentication Mechanisms:** 
  * **Kaggle API:** Authenticated via runtime generation or passing of a secure `kaggle.json` credential file containing API keys.
  * **GCP/GCS Access:** Authenticated interactively within the notebook using the runtime credential provider framework via `google.colab.auth.authenticate_user()`. This verifies organizational IAM permissions without hardcoding secrets.

---

## 🧪 Data Engineering & Methodology

### 6. Data Split & Validation Strategy (Criterion 6)
* **Splitting Protocol:** The data is clean-split into explicit **Train, Dev (Validation), and Test** sets.
* **Leakage Prevention:** 
  * Splits are established sequentially prior to any heavy feature transformation or scaling steps.
  * To strictly avoid **data leakage**, all preprocessing computations (e.g., missing value imputation rules, scaling calculations) are fitted *only* on the training split, and then applied as fixed transformations to the dev and test sets.

### 7. Feature Descriptions (Criterion 7 & 8)
Below are the core features mapped within our machine learning architecture:

| Feature Name | Data Type | Storage Format | Description & Engineering Origin |
| :--- | :--- | :--- | :--- |
| `provider_id` | Categorical | String / Category | Unique hash identifying the healthcare facility or physician. |
| `claim_amount` | Numerical | Float64 | Total dollar value requested by the provider. |
| `procedure_code` | Categorical | Integer / String | Standardized ICD-10/CPT billing identifier code. |
| `is_fraud` | Binary | Integer (0/1) | **Target Variable**. 1 indicates a confirmed fraudulent claim. |

### 8. Reproducibility of Data Collection (Criterion 9)
* **Data Source:** Programmatically pulled from the official **Kaggle API**.
* **Collection Steps:** 
  1. Initialize connection to Kaggle via the execution environment using automated API credentials.
  2. Download the compressed raw archive directly into the local Colab runtime space.
  3. Extract files and stage them to the primary raw Google Cloud Storage repository path.

### 9. Reproducibility of Preprocessing (Criterion 10)
To fully recreate the finalized data splits from the raw source files, our preprocessing module executes the following strict workflow steps:
1. **Data Cleaning:** Eliminates duplicate entries and handles invalid structural anomalies.
2. **Train/Dev/Test Separation:** Splitting is performed upstream to isolate evaluation environments.
3. **Transformations & Scaling:** Encodes categorical variants and standardizes numeric boundaries.
4. **Cloud Serialization:** Simultaneously writes out the finished splits back to GCS as paired `.csv` and optimized `.parquet` targets.

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

4. **Hardware Execution:**
   * Set your runtime type via `Runtime` > `Change runtime type` to **CPU** (standard processing is sufficient for Milestone 1 pipeline construction).
