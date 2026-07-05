# Yes Bank Stock Closing Price Prediction

Regression project predicting the monthly **Closing price** of Yes Bank stock using its historical Open, High, and Low prices — built as an end-to-end EDA + ML capstone project.

## 📌 Project Overview

Yes Bank grew steadily from its 2004 inception until a major fraud case involving co-founder and then-CEO Rana Kapoor came to light in 2018, followed by mounting bad loans and an RBI-imposed moratorium in March 2020. The stock collapsed from an all-time high of ~₹367 (Aug 2018) to under ₹12 (mid-2020).

This project analyzes 185 months of OHLC (Open, High, Low, Close) stock data from **July 2005 to November 2020**, and builds a regression model to predict Close price.

## 📂 Repository Structure

| File | Description |
|---|---|
| `data_YesBank_StockPrices.csv` | Raw monthly OHLC dataset (185 rows) |
| `EDA_YesBank_StockPrices.ipynb` | Full exploratory data analysis — 15 visualizations, correlation heatmap, pair plot, business recommendations |
| `ML_YesBank_StockPrices.ipynb` | Hypothesis testing, feature engineering, and 3 regression models (Linear, Ridge, Random Forest) |
| `yes_bank_ridge_model.pkl` | Final saved model (tuned Ridge Regression) |
| `yes_bank_scaler.pkl` | StandardScaler used to preprocess features |
| `requirements.txt` | Python dependencies |

## 🔍 Key Findings

- Dataset is clean: no missing values, no duplicates
- Open, High, Low, and Close are extremely highly correlated (0.97–0.995)
- The 2019–2020 crisis-period average Close was significantly lower than the 2016–2018 peak (t-test, p ≈ 7.2e-13)
- **Low** is the single strongest predictor of Close (correlation ≈ 0.995)

## 🤖 Model Performance

| Model | Test R² | RMSE (₹) |
|---|---|---|
| Linear Regression | 0.991 | 9.17 |
| **Ridge Regression (tuned) — Final Model** | **0.991** | **9.17** |
| Random Forest (tuned) | 0.980 | 13.37 |

**Final model:** Ridge Regression (alpha = 0.001), chosen for its accuracy, interpretability, and robustness to multicollinearity between Open/High/Low.

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook EDA_YesBank_StockPrices.ipynb
jupyter notebook ML_YesBank_StockPrices.ipynb
```

## 🔮 Load the Saved Model

```python
import pickle

with open('yes_bank_ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('yes_bank_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example: predict Close from Open, High, Low, Year, Month
sample = [[30.2, 32.6, 26.8, 2007, 2]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(f"Predicted Close: ₹{prediction[0]:.2f}")
```

## 👤 Author

**Dnyaneshwar Gumalwad**
BCA (AI/ML), JSPM University, Pune
