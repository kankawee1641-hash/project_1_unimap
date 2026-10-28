# IFJC4002 | Used Car Price Prediction System (UniMAP)
**Course:** IFJC4002 Practical Artificial Intelligence with Python  
**Institution:** Faculty of Intelligent Computing, Universiti Malaysia Perlis (UniMAP)  
**Coordinator:** Ir. Dr. Marni Azira Markom  
**Case Study:** Case 2 - Used Car Price Prediction (Regression Task)

---

## 📌 Project Overview
This project develops an end-to-end Machine Learning prediction system for estimating the resale price of used cars in Malaysia (in Malaysian Ringgit, RM). 

### Target & Features:
* **Target (`y`):** `resale_price_rm` (Continuous value in RM)
* **Numeric Features:**
  - `vehicle_age_years`: Age of the car (0 - 18+ years)
  - `mileage_km`: Total distance driven in kilometers
  - `engine_size_l`: Engine displacement (0.8 - 5.0 L)
  - `condition_rating`: Vehicle condition score (1 = Poor, 5 = Pristine)
  - `previous_owners`: Number of prior owners (0 - 4+)
* **Categorical Features:**
  - `fuel_type`: Petrol, Diesel, Hybrid
  - `transmission`: Automatic, Manual
  - `market_segment`: Economy, Mid_Range, Premium

---

## 📁 Repository & Folder Structure
```text
D:\project_1_unimap\
│
├── data/
│   ├── raw/                 # Original dataset (02_Used_Car_Price.xlsx)
│   └── processed/           # Cleaned and prepared data splits
├── notebooks/
│   └── used_car_price_prediction.ipynb  # Step-by-step EDA & Model comparison notebook
├── src/
│   ├── train.py             # Script for preprocessing, training, and exporting pipeline
│   └── app.py               # Interactive Streamlit dashboard application
├── models/
│   └── car_price_pipeline.joblib # Exported end-to-end model pipeline
├── outputs/
│   └── prediction_logs.csv  # Operational monitoring logs of user inputs and predictions
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation and deployment guide
```

---

## ⚙️ Installation & Local Setup

### 1. Prerequisites
Ensure Python 3.9+ is installed on your Windows machine.

### 2. Install Dependencies
Open **Command Prompt** or **PowerShell** in `D:\project_1_unimap`:
```bash
cd D:\project_1_unimap
pip install -r requirements.txt
```

### 3. Model Training & Export
To train the baseline and regression algorithms (Linear Regression, Ridge, Random Forest, Gradient Boosting) and export the best pipeline:
```bash
python src/train.py
```

### 4. Run Interactive Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run src/app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🚀 Deployment to Streamlit Community Cloud
1. Push your project code (`app.py`, `models/car_price_pipeline.joblib`, `requirements.txt`, and `README.md`) to a GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Select your repository, branch (`main`), and set the main file path to `src/app.py`.
4. Click **Deploy!** Your app will be live with a public URL for grading and presentation.

---

## 🔍 Model Monitoring & Maintenance Trigger
* **Input Anomaly Detection:** The dashboard flags inputs exceeding normal operational boundaries (e.g. Mileage > 450,000 km or Vehicle Age > 18 years).
* **Monitoring Logs:** Predictions and timestamps are stored in `outputs/prediction_logs.csv` for audit trails.
* **Review Trigger:** If more than 15% of logged requests are out-of-distribution, or when macroeconomic vehicle taxes/depreciation policies change, trigger a model retraining cycle.
