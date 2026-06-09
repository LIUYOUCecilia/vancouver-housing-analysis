import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

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
    raw_path = os.path.join(project_dir, "raw_data", "vancouver_housing.csv")
    df_raw = pd.read_csv(raw_path)
    
    # 2. Plotting: Outliers distribution comparison
    print("Generating outlier distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df_raw["price_cad"] / 1e6, bins=40, kde=True, ax=axes[0], color="#f43f5e")
    axes[0].set_title("Price Distribution Before Outlier Removal (with Outliers)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Price (Millions CAD)", fontsize=10)
    axes[0].set_ylabel("Frequency", fontsize=10)
    
    sns.histplot(df_cleaned["price_cad"] / 1e6, bins=40, kde=True, ax=axes[1], color="#10b981")
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
    X = df_cleaned[["distance_to_beach_km", "precipitation_mm", "bedrooms"]]
    
    # Add constant for intercept term beta_0
    X_with_const = sm.add_constant(X)
    
    # Fit OLS
    model = sm.OLS(Y, X_with_const)
    results = model.fit()
    
    # Save OLS text summary
    summary_path = os.path.join(assets_dir, "regression_results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(results.summary().as_text())
    print(f"  Saved OLS text summary to {summary_path}")
    
    # Print OLS summary to stdout
    print("\n" + "="*80)
    print("OLS MULTIPLE LINEAR REGRESSION SUMMARY RESULTS")
    print("="*80)
    print(results.summary())
    print("="*80 + "\n")
    
    # 4. Plotting: Price vs. Distance to Beach with regression line
    print("Generating regression fit plot...")
    plt.figure(figsize=(10, 6))
    
    # We plot the data points colored by number of bedrooms
    scatter = plt.scatter(
        df_cleaned["distance_to_beach_km"], 
        df_cleaned["price_cad"] / 1e6, 
        c=df_cleaned["bedrooms"], 
        cmap="viridis", 
        alpha=0.6, 
        edgecolors="none",
        s=35
    )
    cb = plt.colorbar(scatter)
    cb.set_label("Number of Bedrooms", fontsize=10)
    
    # Fit line of Price vs Distance to beach (ignoring other features for visual simplicity)
    b_beach = results.params["distance_to_beach_km"]
    intercept = results.params["const"]
    # Mean of other features to project a 2D line
    mean_beds = df_cleaned["bedrooms"].mean()
    mean_precip = df_cleaned["precipitation_mm"].mean()
    b_beds = results.params["bedrooms"]
    b_precip = results.params["precipitation_mm"]
    
    x_range = np.linspace(df_cleaned["distance_to_beach_km"].min(), df_cleaned["distance_to_beach_km"].max(), 100)
    y_pred_line = (intercept + b_beach * x_range + b_beds * mean_beds + b_precip * mean_precip) / 1e6
    
    plt.plot(x_range, y_pred_line, color="#e11d48", linewidth=2.5, label="Fitted Regression Line (at Mean Beds/Precip)")
    plt.title("Vancouver House Price vs. Distance to Beach (with Regression Fit)", fontsize=13, fontweight="bold")
    plt.xlabel("Distance to Beach (km)", fontsize=11)
    plt.ylabel("Price (Millions CAD)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    
    fit_plot_path = os.path.join(assets_dir, "regression_fit_beach.png")
    plt.savefig(fit_plot_path, dpi=300)
    plt.close()
    print(f"  Saved regression fit plot to {fit_plot_path}")
    
    # 5. Plotting: Residual Plot
    print("Generating residual diagnostic plot...")
    plt.figure(figsize=(10, 6))
    
    fitted_vals = results.fittedvalues
    residuals = results.resid
    
    sns.scatterplot(x=fitted_vals / 1e6, y=residuals / 1e3, alpha=0.5, color="#5b21b6")
    plt.axhline(y=0, color="#ef4444", linestyle="--", linewidth=2)
    plt.title("Residual vs. Fitted Values Plot (Assumption Check)", fontsize=13, fontweight="bold")
    plt.xlabel("Fitted Values (Millions CAD)", fontsize=11)
    plt.ylabel("Residuals (Thousands CAD)", fontsize=11)
    
    residual_plot_path = os.path.join(assets_dir, "residual_diagnostic.png")
    plt.savefig(residual_plot_path, dpi=300)
    plt.close()
    print(f"  Saved residual diagnostic plot to {residual_plot_path}")
    
    # 6. Simple interpretation outputs
    print("Statistical Interpretations:")
    print(f"  - R-squared: {results.rsquared:.4f} (The model explains {results.rsquared*100:.2f}% of price variance)")
    print(f"  - Intercept (Base Price): {results.params['const']:,.2f} CAD")
    print(f"  - Beach distance coefficient: {results.params['distance_to_beach_km']:,.2f} CAD (For every 1km further from the beach, price decreases by {-results.params['distance_to_beach_km']:,.2f} CAD, p-value: {results.pvalues['distance_to_beach_km']:.4e})")
    print(f"  - Monthly rainfall coefficient: {results.params['precipitation_mm']:,.2f} CAD (For every 1mm of monthly rain, price decreases by {-results.params['precipitation_mm']:,.2f} CAD, p-value: {results.pvalues['precipitation_mm']:.4e})")
    print(f"  - Bedrooms coefficient: {results.params['bedrooms']:,.2f} CAD (For each additional bedroom, price increases by {results.params['bedrooms']:,.2f} CAD, p-value: {results.pvalues['bedrooms']:.4e})")
    
    return results

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    run_regression_analysis(project_dir)
