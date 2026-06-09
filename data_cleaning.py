import os
import pandas as pd

def clean_and_merge_data(project_dir):
    raw_dir = os.path.join(project_dir, "raw_data")
    processed_dir = os.path.join(project_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    housing_path = os.path.join(raw_dir, "vancouver_housing.csv")
    climate_path = os.path.join(raw_dir, "vancouver_climate.csv")
    
    if not os.path.exists(housing_path) or not os.path.exists(climate_path):
        raise FileNotFoundError("Raw datasets not found! Run generate_data.py first.")
        
    print("Loading datasets...")
    df_housing = pd.read_csv(housing_path)
    df_climate = pd.read_csv(climate_path)
    
    # 1. Map housing transaction dates to year_month (YYYY-MM)
    print("Mapping transaction dates to Year-Month...")
    df_housing["year_month"] = pd.to_datetime(df_housing["date"]).dt.strftime("%Y-%m")
    
    # 2. Perform merge (Inner join)
    print("Merging housing transactions with monthly climate data...")
    df_combined = pd.merge(df_housing, df_climate, on="year_month", how="left")
    
    # 3. Detect and remove outliers in price_cad using 1.5 * IQR rule
    print("Detecting and filtering pricing outliers using 1.5 * IQR rule...")
    price_col = df_combined["price_cad"]
    
    q1 = price_col.quantile(0.25)
    q3 = price_col.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    print(f"  Q1 (25th percentile): {q1:,.2f} CAD")
    print(f"  Q3 (75th percentile): {q3:,.2f} CAD")
    print(f"  IQR (Interquartile Range): {iqr:,.2f} CAD")
    print(f"  Lower Bound: {lower_bound:,.2f} CAD")
    print(f"  Upper Bound: {upper_bound:,.2f} CAD")
    
    # Identify outliers
    outliers = df_combined[(price_col < lower_bound) | (price_col > upper_bound)]
    df_cleaned = df_combined[(price_col >= lower_bound) & (price_col <= upper_bound)]
    
    num_outliers = len(outliers)
    print(f"  Outliers identified: {num_outliers} records out of {len(df_combined)} total.")
    print(f"  Average price of outliers: {outliers['price_cad'].mean():,.2f} CAD")
    
    # Save cleaned combined dataset
    cleaned_path = os.path.join(processed_dir, "vancouver_combined_cleaned.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"Cleaned and merged dataset saved to {cleaned_path} ({len(df_cleaned)} records)")
    
    return df_cleaned, outliers

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    clean_and_merge_data(project_dir)
