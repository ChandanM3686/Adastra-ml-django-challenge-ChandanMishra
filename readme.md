# AdAstraa AI – ML + Django Challenge  
### Predicting Sale_Amount from Real-World Marketing Data  
**Developer:** Chandan Mishra  

---

## 📌 Project Overview
This project is built as part of the **24-hour Machine Learning Engineer (Full-Stack)** challenge by **AdAstraa AI**.

The Django web application:

- Trains a Machine Learning model (CatBoost Regressor)
- Cleans messy real-world advertising data
- Stores the trained model inside the Django backend
- Lets users upload a `test.csv` **without Sale_Amount**
- Applies the same preprocessing pipeline used during training
- Generates a downloadable `submission.csv` with:
  - All original columns
  - A new column: **Predicted_Sale_Amount**

This simulates real-world eCommerce marketing analytics challenges.

---

## 📁 Dataset Details
The dataset contains intentionally messy fields:

- Mixed date formats (`YYYY/MM/DD`, `DD-MM-YYYY`, `MM/DD/YY`)
- Inconsistent strings (typos, casing differences)
- Values like `"1.2k"`, `"2M"`, `"Rs. 1200"`, `"1,200"`
- Missing values and incorrect “Conversion Rate”
- Noise in categorical fields like “Campaign_Name”, “Keyword”, “Location”

---

## 🧼 1️⃣ Data Cleaning & Feature Engineering

### ✔ Numeric Cleaning
- Converted shorthand values like:
  - `"1.2k"` → `1200`
  - `"3M"` → `3,000,000`
- Cleaned cost values containing symbols (`₹`, `$`, `Rs.`)
- Removed unwanted characters using regex
- Filled missing numeric values with **train medians**

### ✔ Monetary Parsing
Handled:
- `(1200)` → `-1200`
- `"1,200.50"` → `1200.50`
- `"$1.2k"` → `1200`

### ✔ Date Normalization
Parsed mixed formats and extracted:
- `ad_year`
- `ad_month`
- `ad_day`
- `ad_weekday`

### ✔ Categorical Handling
- Lowercased all values
- Handled inconsistencies like:
  - `"Mobile"` & `"mobile"`
  - `"New York"` & `"new-york"` & `"NewYork"`
- Replaced missing values with `"missing"`

### ✔ Feature Engineering
Created additional features:

- `Conversion Rate` (fixed incorrect values)
- `ctr = Clicks / Impressions`
- `cost_per_click = Cost / Clicks`

---

## 🤖 2️⃣ Modeling Approach

The chosen model: **CatBoost Regressor**

### ✔ Why CatBoost?
- Handles messy categorical data efficiently  
- Robust to missing values  
- Performs well with noisy real-world data  
- No extensive manual encoding required  
- Built-in handling of overfitting via **early stopping**

### ✔ Model Setup
- `iterations=2000`
- `learning_rate=0.03`
- `depth=6`
- Early stopping after 100 rounds  
- Automatic best-iteration selection

### ✔ Validation Metrics
Using validation set (15% split):

| Metric | Value |
|--------|--------|
| RMSE | ~290.42 |
| MAPE | ~0.174 |
| Approx Accuracy | **82.5%** |

---

## 🏗 3️⃣ Django Application Flow

### ✔ User Flow
1. Open the app (`/`)
2. Upload:
   - `train.csv` (optional)
   - `test.csv` (optional)
3. The system:
   - Cleans + preprocesses data
   - Trains CatBoost model
   - Saves the model (`final_model.joblib`)
   - Generates predictions for `test.csv`
4. Click the **Download submission.csv** button

### ✔ Backend Architecture
