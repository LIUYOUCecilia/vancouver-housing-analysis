import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Set style for visuals
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Import local data cleaning function
from data_cleaning import clean_and_merge_data

def run_regression_analysis(project_dir):
    assets_dir = os.path.join(project_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # 1. Load and clean datasets
    df_cleaned, df_outliers = clean_and_merge_data(project_dir)
    
    raw_path = os.path.join(project_dir, "raw_data", "vancouver_combined_real.csv")
    df_raw = pd.read_csv(raw_path)
    # Calculate raw price for raw dataset
    df_raw['price_cad'] = df_raw['current_land_value'] + df_raw['current_improvement_value'].fillna(0)
    
    # 2. Plotting: Outliers distribution comparison
    print("Generating outlier distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pre-cleaning distribution (excluding properties with zero or null price)
    df_raw_valid = df_raw[df_raw["price_cad"] > 0]
    sns.histplot(df_raw_valid["price_cad"] / 1e6, bins=50, kde=True, ax=axes[0], color="#f43f5e")
    axes[0].set_title("Price Distribution Before Outlier Removal (with Outliers)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Price (Millions CAD)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    
    sns.histplot(df_cleaned["price_cad"] / 1e6, bins=50, kde=True, ax=axes[1], color="#10b981")
    axes[1].set_title("Price Distribution After 1.5 * IQR Outlier Removal", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Price (Millions CAD)", fontsize=10)
    axes[1].set_ylabel("Frequency", fontsize=10)
    
    plt.tight_layout()
    dist_plot_path = os.path.join(assets_dir, "outlier_removal_comparison.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"  Saved outlier comparison plot to {dist_plot_path}")
    
    # 3. Fit Multiple Linear Regression Model
    print("Fitting OLS Regression Model...")
    
    # Define dependent (Y) and independent variables (X)
    Y = df_cleaned["price_cad"]
    X = df_cleaned[["distance_to_beach_km", "annual_precip_mm", "age_at_assessment", "is_strata"]]
    
    # Add constant for intercept term beta_0
    X_with_const = sm.add_constant(X)
    
    # Fit OLS
    model = sm.OLS(Y, X_with_const)
    results = model.fit()
    
    # 4. Rigorous Diagnostic Checks
    print("Running Model Diagnostics...")
    
    # 4.1. Multicollinearity (VIF)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data.to_string(index=False))
    
    # 4.2. Heteroscedasticity (Breusch-Pagan)
    bp_test = het_breuschpagan(results.resid, X_with_const)
    bp_labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_results = dict(zip(bp_labels, bp_test))
    print("\nBreusch-Pagan Test Results:")
    for key, value in bp_results.items():
        print(f"  {key}: {value:.6f}")
        
    # Save diagnostics and summary text
    summary_path = os.path.join(assets_dir, "regression_results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("OLS MULTIPLE LINEAR REGRESSION SUMMARY RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(results.summary().as_text())
        f.write("\n\n" + "="*80 + "\n")
        f.write("MODEL DIAGNOSTICS\n")
        f.write("="*80 + "\n\n")
        f.write("1. Multicollinearity (VIF):\n")
        f.write(vif_data.to_string(index=False))
        f.write("\n\n2. Heteroscedasticity (Breusch-Pagan Test):\n")
        for key, value in bp_results.items():
            f.write(f"   {key}: {value:.6e}\n")
    print(f"\n  Saved OLS text summary and diagnostics to {summary_path}")
    
    # Print OLS summary to stdout
    print("\n" + "="*80)
    print("OLS MULTIPLE LINEAR REGRESSION SUMMARY RESULTS")
    print("="*80)
    print(results.summary())
    print("="*80 + "\n")
    
    # 5. Plotting: Price vs. Distance to Beach with regression line
    print("Generating regression fit plot...")
    plt.figure(figsize=(10, 6))
    
    # We plot the data points colored by property age
    scatter = plt.scatter(
        df_cleaned["distance_to_beach_km"], 
        df_cleaned["price_cad"] / 1e6, 
        c=df_cleaned["age_at_assessment"], 
        cmap="viridis_r", 
        alpha=0.6, 
        edgecolors="none",
        s=35
    )
    cb = plt.colorbar(scatter)
    cb.set_label("Property Age (Years)", fontsize=10)
    
    # Fit line of Price vs Distance to beach (projecting at means of other features)
    b_beach = results.params["distance_to_beach_km"]
    intercept = results.params["const"]
    mean_precip = df_cleaned["annual_precip_mm"].mean()
    mean_age = df_cleaned["age_at_assessment"].mean()
    mean_strata = df_cleaned["is_strata"].mean()
    
    b_precip = results.params["annual_precip_mm"]
    b_age = results.params["age_at_assessment"]
    b_strata = results.params["is_strata"]
    
    x_range = np.linspace(df_cleaned["distance_to_beach_km"].min(), df_cleaned["distance_to_beach_km"].max(), 100)
    y_pred_line = (intercept + b_beach * x_range + b_precip * mean_precip + b_age * mean_age + b_strata * mean_strata) / 1e6
    
    plt.plot(x_range, y_pred_line, color="#e11d48", linewidth=2.5, label="Fitted OLS Line (at Mean Age/Precip/Strata)")
    plt.title("Vancouver Property Value vs. Distance to Beach (with OLS Fit)", fontsize=13, fontweight="bold")
    plt.xlabel("Distance to Beach (km)", fontsize=11)
    plt.ylabel("Property Assessed Value (Millions CAD)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    
    fit_plot_path = os.path.join(assets_dir, "regression_fit_beach.png")
    plt.savefig(fit_plot_path, dpi=300)
    plt.close()
    print(f"  Saved regression fit plot to {fit_plot_path}")
    
    # 6. Plotting: Residual Plot (Fitted vs Residuals)
    print("Generating residual diagnostic plot...")
    plt.figure(figsize=(10, 6))
    
    fitted_vals = results.fittedvalues
    residuals = results.resid
    
    sns.scatterplot(x=fitted_vals / 1e6, y=residuals / 1e3, alpha=0.5, color="#5b21b6")
    plt.axhline(y=0, color="#ef4444", linestyle="--", linewidth=2)
    plt.title("Residual vs. Fitted Values Plot (Heteroscedasticity Check)", fontsize=13, fontweight="bold")
    plt.xlabel("Fitted Values (Millions CAD)", fontsize=11)
    plt.ylabel("Residuals (Thousands CAD)", fontsize=11)
    
    residual_plot_path = os.path.join(assets_dir, "residual_diagnostic.png")
    plt.savefig(residual_plot_path, dpi=300)
    plt.close()
    print(f"  Saved residual diagnostic plot to {residual_plot_path}")
    
    # 7. Plotting: Q-Q Plot of Residuals (Normality Check)
    print("Generating Q-Q Plot of residuals...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sm.qqplot(residuals, line='s', ax=ax, color="#2563eb", alpha=0.5)
    ax.set_title("Normal Q-Q Plot of Residuals (Normality Check)", fontsize=13, fontweight="bold")
    ax.get_lines()[1].set_color("#dc2626")  # Change reference line to red
    ax.get_lines()[1].set_linewidth(2)
    
    qq_plot_path = os.path.join(assets_dir, "residual_qq_plot.png")
    plt.savefig(qq_plot_path, dpi=300)
    plt.close()
    print(f"  Saved Q-Q plot to {qq_plot_path}")
    
    return results

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    run_regression_analysis(project_dir)
