# 📊 Dataset Name: Healthcare Fraud Detection Dataset

**Short Description:** A highly realistic synthetic dataset designed to replicate real-world medical billing patterns and fraudulent behavior. It is curated to help build, test, and evaluate predictive machine learning classification models and anomaly detection systems.

---

## 📌 Quick Metadata
* **Version:** 1.0.0
* **Format:** CSV (1 File)
* **Size:** ~288 KB | 10,000 synthetic insurance claim records
* **License:** CC0: Public Domain (Simulated data ensuring complete privacy compliance)
* **Status:** Static / Ready for Analysis

---

## 📂 Source & Collection
* **Origin:** Curated and uploaded to Kaggle by Data Scientist Nudrat Abbas. The data structure relies on a log-normal distribution to mimic actual medical billing anomalies without using PII.
* **Collection Date:** Published/Updated in 2026.
* **Maintainer:** Nudrat Abbas (via Kaggle).

---

## 🧭 Schema & Data Dictionary

The dataset aggregates comprehensive features across patient demographics, provider metrics, clinical coding, and financials. Below is the complete mapping of all **20 columns**:

| Column Name | Data Type | Description | Sample/Expected Values |
| :--- | :--- | :--- | :--- |
| `Provider_ID` | `object` | Unique alphanumeric identifier for the healthcare provider | `PROV10234` |
| `Claim_ID` | `object` | Unique identifier assigned to the specific insurance claim | `CLM992381` |
| `Patient_Age` | `int64` | The age of the patient in years | `45`, `67` |
| `Patient_Gender` | `object` | Categorical gender indicator of the patient | `Male`, `Female` |
| `Diagnosis_Code` | `object` | Alphanumeric medical code representing the diagnosis (ICD-10 style) | `Essential hypertension` |
| `Procedure_Code` | `int64` | Numeric billing identifier for the medical procedure performed (CPT style) | `99213` |
| `Claim_Amount` | `float64` | The total dollar amount billed by the provider to the insurance | `1250.50` |
| `Approved_Amount` | `float64` | The actual dollar amount approved and paid out by the insurance | `850.00` |
| `Insurance_Type` | `object` | The insurance payer category | `Private`, `Medicare`, `Medicaid`, `Self-Pay` |
| `Claim_Submission_Date` | `object` | The date the insurance claim was submitted (requires datetime parsing) | `YYYY-MM-DD` |
| `Days_Between_Service_and_Claim`| `int64` | The temporal gap (in days) between the treatment date and filing date | `5`, `14` |
| `Number_of_Claims_Per_Provider_Monthly` | `int64` | The numeric volume of claims pushed by this specific provider in a month | `142` |
| `Provider_Specialty` | `object` | Medical clinical field of the practicing provider | `Cardiology`, `General Practice` |
| `Patient_State` | `object` | The state/geographic location of the patient | `CA`, `NY`, `TX` |
| `Claim_Status` | `object` | The settlement outcome/state of the filed claim transaction | `Approved`, `Pending`, `Rejected` |
| `Is_Fraud` | `int64` | **Target Variable.** Flag indicating if a claim is illicit (`0` = Clean, `1` = Fraudulent) | `0`, `1` |
| `Length_of_Stay` | `int64` | Total duration of the patient's clinic/hospital encounter in days | `0` (Outpatient), `3` |
| `Visit_Type` | `object` | Setting type where medical services were rendered | `Inpatient`, `Outpatient`, `Emergency` |
| `Chronic_Condition_Flag` | `int64` | Boolean-style indicator if the patient has a diagnosed underlying illness | `0` (No), `1` (Yes) |
| `Prior_Visits_12m` | `float64` | Cumulative number of medical encounters the patient had over the past year | `2.0`, `0.0` |

---

## ⚠️ Known Limitations & Cleaning Steps
* **Class Imbalance:** Fraudulent instances typically represent a minority class in healthcare claims (often around ~8%). Your modeling pipeline will likely require resampling techniques (like SMOTE) or class-weighted evaluation metrics (Precision-Recall AUC instead of pure accuracy).
* **Date Parsing:** `Claim_Submission_Date` is typed as an `object` (string). It needs to be converted to a true datetime format in pandas before extracting temporal trends.
* **Synthetic Nature:** While it mirrors clinical distributions and billing anomalies perfectly, it is artificial data meant strictly for research and benchmarking.

---

## 🚀 Usage Example

Below is a Python baseline snippet to read the file out of your project's local directory layout and examine the fraud target split:

```python
import pandas as pd

# Load the dataset out of your local directory structure
df = pd.read_csv('data/healthcare_fraud_detection_dataset.csv')

# Properly parse your date column right away
df['Claim_Submission_Date'] = pd.to_datetime(df['Claim_Submission_Date'])

# Inspect the shape and fraud imbalance
print("Dataset Shape:", df.shape)
print("\nTarget Class Distribution:")
print(df['Is_Fraud'].value_counts(normalize=True) * 100)
```