# Vancouver Real Estate & Climate Regression Analysis (Dual-Path)

**Live Streamlit App:** [liuyoucecilia-vancouver-housing-analysis-app-l8nug5.streamlit.app](https://liuyoucecilia-vancouver-housing-analysis-app-l8nug5.streamlit.app/)

This repository contains an advanced data science project implementing a **Dual-Path** (双线并行) optimization plan for real estate valuation and climate regression analysis. The project bridges academic econometrics and interactive product development by:

1. **Academic/Research Path (UBC RA Prep)**: Utilizing real Vancouver property tax assessments (City of Vancouver Open Data) and real YVR Airport climate observations (Environment Canada/Meteostat WMO: `71892`), running multiple linear regression with VIF and Breusch-Pagan diagnostic tests, and implementing Stanford CS229 Normal Equation matrix math validation.
2. **Fiverr/Upwork Showcase Path**: Wrapping the fitted OLS regression coefficients into a live, interactive Streamlit web dashboard with Plotly 3D visuals and dynamic prediction sliders.

---

## Key Project Components

1. **Real Data Harvester (`fetch_real_data.py`)**: 
   - Downloads daily temperature and precipitation records from 2015-2024 via the `meteostat` API.
   - Harvests ~5,000 property tax assessment records from the Vancouver Open Data Portal.
   - Calculates geographic distance from each property's local centroid to the nearest Vancouver beach (Kitsilano, English Bay, Spanish Banks) using the Haversine formula.
   - Merges property assessments with previous-year annual climate metrics to represent lag in property assessment cycles.
2. **Data Cleaning (`data_cleaning.py`)**: 
   - Computes total property assessed value (`price_cad`), building age, and encodes property class (`is_strata`).
   - Filters out extreme valuation outliers using the standard **1.5 * IQR** rule.
3. **Econometric Regression & Diagnostics (`regression_analysis.py`)**: 
   - Fits OLS multiple linear regression using `statsmodels`.
   - Computes **Variance Inflation Factors (VIF)** to validate multicollinearity.
   - Runs **Breusch-Pagan Lagrange Multiplier tests** to diagnose residual heteroscedasticity.
4. **OLS Normal Equation Solver (`normal_equation_validation.py`)**: 
   - Solves $\hat{\beta} = (X^T X)^{-1} X^T Y$ using NumPy matrix multiplication and validates that coefficients match statsmodels parameters perfectly.
5. **Streamlit Interactive Dashboard (`app.py`)**: 
   - A modern web app allowing users to input property features (sliders) and get real-time price predictions, with Plotly 3D visual models and diagnostic summaries.

---

## Regression Model Results

The OLS model was fitted on **4,326 cleaned records** (with 270 outliers filtered out by the 1.5 * IQR rule). It yields an **Adjusted $R^2$ of 0.583**, indicating that **58.3% of the variance** in Vancouver property values is explained by the model.

### OLS Coefficients & Significance ($\alpha = 0.05$)

| Feature | Coefficient ($\beta$) | t-statistic | p-value | Interpretation & Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Intercept ($\beta_0$)** | +$8,268,302.71 | 13.90 | < 0.001 | Baseline valuation (Highly Significant) |
| **Beach Distance ($\beta_1$)** | -$79,328.86 | -20.12 | < 0.001 | Price decreases by **$79,328.86 CAD** per km of distance from the beach. (Highly Significant) |
| **Precipitation ($\beta_2$)** | -$4,771.87 | -8.46 | < 0.001 | Price decreases by **$4,771.87 CAD** per mm of annual precipitation in preceding year. (Highly Significant) |
| **Building Age ($\beta_3$)** | -$6,047.41 | -14.86 | < 0.001 | Property depreciates by **$6,047.41 CAD** per year of age. (Highly Significant) |
| **Is Strata ($\beta_4$)** | -$1,878,340.86 | -69.44 | < 0.001 | Strata units (condos/townhouses) are assessed **$1.88M CAD** lower than single-family land estates. (Highly Significant) |

### Academic Diagnostics & Discussions
* **Multicollinearity (VIF)**: The VIF for `annual_precip_mm` is **16.03** (exceeding the warning threshold of 10). This collinearity is a natural result of pooling property-level micro data with annual-level macro climate observations. Standard errors are inflated but OLS coefficients remain unbiased.
* **Heteroscedasticity (Breusch-Pagan)**: The BP test rejects homoscedasticity ($LM = 160.09, p < 0.001$). This is typical in housing markets where residual variance scales with absolute property value. Standard errors can be adjusted using White's heteroscedasticity-robust covariance matrix (HC1/HC3).

---

## Normal Equation Validation

Using NumPy, we validated the analytical solution of OLS:

$$\hat{\beta} = (X^T X)^{-1} X^T Y$$

The manual solver matches `statsmodels` parameters perfectly:
* **NumPy Solver**: `[8268302.70830, -79328.86478, -4771.87070, -6047.41262, -1878340.86005]`
* **Statsmodels**: `[8268302.70830, -79328.86478, -4771.87070, -6047.41262, -1878340.86005]`
* **Status**:  **Verified Perfect Match (Difference < 1e-10)**

---

## How to Run the Pipeline & App

### 1. Prerequisites
Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Fetch and Analyze Data
Harvest raw data and run regression modeling:
```bash
# 1. Fetch real weather and property tax datasets
python fetch_real_data.py

# 2. Run data cleaning and regression modeling (generates plots in assets/)
python regression_analysis.py

# 3. Perform manual OLS matrix math validation
python normal_equation_validation.py
```

### 3. Launch Streamlit Web App
Run the interactive dashboard locally:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud (Live Online App)

You can deploy this interactive dashboard to the web for free:
1. Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository: `vancouver-housing-analysis`.
4. Set **Main file path** to `app.py`.
5. Click **Deploy!**

Once deployed, Streamlit will provide a public URL. You can add this live URL to the "About" website link on your GitHub repository page so visitors can interact with it instantly.

Current app URL: [https://liuyoucecilia-vancouver-housing-analysis-app-l8nug5.streamlit.app/](https://liuyoucecilia-vancouver-housing-analysis-app-l8nug5.streamlit.app/)
