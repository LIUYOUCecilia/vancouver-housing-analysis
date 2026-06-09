# Vancouver Real Estate & Climate Regression Analysis 🏡🌧️

This repository contains a data science portfolio project analyzing the statistical dependencies of residential property prices in Vancouver, BC. It explores how a home's structural traits (bedrooms), proximity to urban amenities (distance to the beach), and seasonal climate patterns (monthly precipitation) impact housing valuation.

Additionally, this project includes a mathematical verification script written for **Stanford CS229 (Machine Learning)**, implementing and validating the OLS Normal Equation step-by-step using matrix operations.

---

## 🎯 Key Project Components

1. **Data Synthesis (`generate_data.py`)**: Simulates 1,200 housing transaction records (2015-2025) and 132 monthly climate records with realistic statistical correlations and injected pricing typos/outliers.
2. **Robust Data Prep (`data_cleaning.py`)**: Combines transaction and climate metrics on transaction month and filters pricing outliers using the standard **1.5 * IQR** rule.
3. **Multiple Linear Regression (`regression_analysis.py`)**: Uses `statsmodels` to fit an OLS regression model and evaluate the significance of predictors.
4. **OLS Normal Equation Solver (`normal_equation_validation.py`)**: Manually calculates $\hat{\beta} = (X^T X)^{-1} X^T Y$ using NumPy matrix multiplication and compares it with statsmodels parameters.
5. **Interactive Report (`vancouver_housing_analysis.ipynb`)**: A complete, step-by-step Jupyter Notebook containing the full workflow, diagnostics, and plots.

---

## 📊 Regression Model Results

The OLS model was fitted on **1,195 cleaned records** (with 5 outliers filtered out by the 1.5 * IQR rule). It yields an **Adjusted $R^2$ of 0.819**, indicating that **81.9% of the variance** in Vancouver house prices is explained by the model.

### OLS Coefficients & Significance ($\alpha = 0.05$)

| Feature | Coefficient ($\beta$) | t-statistic | p-value | Interpretation & Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Intercept ($\beta_0$)** | +$655,307.21 | 37.05 | < 0.001 | Base price of a property. (Highly Significant) |
| **Beach Distance ($\beta_1$)** | -$68,835.37 | -59.40 | < 0.001 | For every 1 km further from the beach, price decreases by **$68,835.37 CAD**. (Highly Significant) |
| **Bedrooms ($\beta_3$)** | +$193,148.63 | 43.75 | < 0.001 | Each additional bedroom increases the price by **$193,148.63 CAD**. (Highly Significant) |
| **Precipitation ($\beta_2$)** | -$73.64 | -0.96 | **0.337** | Every 1 mm of rain in the transaction month decreases price by **$73.64 CAD**. (**Not Statistically Significant**) |

> [!NOTE]
> **Key Insight**: While distance to the beach and bedrooms are major drivers of house prices, seasonal monthly precipitation is **not statistically significant** ($p = 0.337 > 0.05$). This indicates that house transaction prices in Vancouver are not directly affected by winter rain, reflecting realistic market behaviors where long-term location and size factors dominate seasonal weather variations.

---

## 🧮 Normal Equation Validation

To verify the analytical solution of Ordinary Least Squares, we computed the OLS estimator using the design matrix $X$ (with ones appended for the intercept) and target vector $Y$:

$$\hat{\beta} = (X^T X)^{-1} X^T Y$$

The manual NumPy matrix solver returned the exact same results as `statsmodels.OLS`:
* **NumPy Solver**: `[655307.21222, -68835.36745, -73.64275, 193148.62537]`
* **Statsmodels OLS**: `[655307.21222, -68835.36745, -73.64275, 193148.62537]`
* **Status**: ✅ **Verified Perfect Match (Difference < 1e-11)**

---

## 🚀 How to Run the Pipeline

### 1. Prerequisites
Ensure you have the required scientific libraries installed:
```bash
pip install numpy pandas matplotlib seaborn statsmodels
```

### 2. Run Scripts
Generate the datasets and run the OLS regression and validation scripts:
```bash
# 1. Synthesize the raw datasets
python generate_data.py

# 2. Run the cleaning and OLS regression analysis
python regression_analysis.py

# 3. Perform manual matrix math verification
python normal_equation_validation.py
```

All generated charts (outlier comparison, regression lines, and residual diagnostic plots) will be saved in the `assets/` folder.
