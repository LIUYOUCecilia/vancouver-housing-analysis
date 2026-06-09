import os
import pandas as pd

def clean_and_merge_data(project_dir):
    raw_dir = os.path.join(project_dir, "raw_data")
    processed_dir = os.path.join(project_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)
    
    combined_path = os.path.join(raw_dir, "vancouver_combined_real.csv")
    
    if not os.path.exists(combined_path):
        raise FileNotFoundError("Merged raw dataset not found! Run fetch_real_data.py first.")
        
    print("Loading raw merged dataset...")
    df_raw = pd.read_csv(combined_path)
    
    # 1. Clean missing values and extract features
    print("Cleaning data and extracting features...")
    
    # Drop rows with critical missing values
    df_cleaned = df_raw.dropna(subset=[
        'current_land_value', 
        'year_built', 
        'tax_assessment_year', 
        'distance_to_beach_km',
        'annual_precip_mm',
        'annual_temp_c'
    ]).copy()
    
    # Make sure they are correct types
    df_cleaned['current_land_value'] = df_cleaned['current_land_value'].astype(float)
    df_cleaned['current_improvement_value'] = df_cleaned['current_improvement_value'].fillna(0).astype(float)
    df_cleaned['year_built'] = df_cleaned['year_built'].astype(int)
    df_cleaned['tax_assessment_year'] = df_cleaned['tax_assessment_year'].astype(int)
    
    # Filter out invalid years (year_built must be <= tax_assessment_year and > 1800)
    df_cleaned = df_cleaned[
        (df_cleaned['year_built'] <= df_cleaned['tax_assessment_year']) & 
        (df_cleaned['year_built'] > 1800)
    ]
    
    # Calculate price (assessed value) and age
    df_cleaned['price_cad'] = df_cleaned['current_land_value'] + df_cleaned['current_improvement_value']
    df_cleaned['age_at_assessment'] = df_cleaned['tax_assessment_year'] - df_cleaned['year_built']
    
    # Encode strata (condo/townhouse) vs land (single-family)
    df_cleaned['is_strata'] = df_cleaned['legal_type'].apply(lambda x: 1 if str(x).upper() == 'STRATA' else 0)
    
    # Keep only relevant columns for OLS
    columns_to_keep = [
        'tax_assessment_year',
        'neighbourhood_name',
        'distance_to_beach_km',
        'age_at_assessment',
        'is_strata',
        'annual_precip_mm',
        'annual_temp_c',
        'price_cad'
    ]
    df_cleaned = df_cleaned[columns_to_keep]
    
    # 2. Detect and remove outliers in price_cad using 1.5 * IQR rule
    print("Detecting and filtering pricing outliers using 1.5 * IQR rule...")
    price_col = df_cleaned["price_cad"]
    
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
    outliers = df_cleaned[(price_col < lower_bound) | (price_col > upper_bound)]
    df_final = df_cleaned[(price_col >= lower_bound) & (price_col <= upper_bound)]
    
    num_outliers = len(outliers)
    print(f"  Outliers identified: {num_outliers} records out of {len(df_cleaned)} total.")
    
    # Save cleaned combined dataset
    processed_path = os.path.join(processed_dir, "vancouver_combined_cleaned.csv")
    df_final.to_csv(processed_path, index=False)
    print(f"Cleaned dataset saved to {processed_path} ({len(df_final)} records)")
    
    return df_final, outliers

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    clean_and_merge_data(project_dir)
