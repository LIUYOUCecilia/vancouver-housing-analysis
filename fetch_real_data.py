import os
import requests
import json
import pandas as pd
from datetime import datetime
import meteostat
import numpy as np

def fetch_real_climate_data(output_dir):
    print("Fetching real climate data from Meteostat for YVR (WMO: 71892)...")
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Fetch daily weather observations
    daily_data = meteostat.daily('71892', start_date, end_date)
    df_daily = daily_data.fetch()
    
    if df_daily.empty:
        raise ValueError("Failed to retrieve weather data from Meteostat.")
        
    print(f"Retrieved {len(df_daily)} daily weather records.")
    
    # Extract year and month from the index
    df_daily['year'] = df_daily.index.year
    df_daily['year_month'] = df_daily.index.strftime('%Y-%m')
    
    # 1. Aggregate to monthly data
    df_monthly = df_daily.groupby('year_month').agg(
        precipitation_mm=('prcp', 'sum'),
        avg_temp_c=('temp', 'mean')
    ).reset_index()
    
    # Round columns
    df_monthly['precipitation_mm'] = df_monthly['precipitation_mm'].round(1)
    df_monthly['avg_temp_c'] = df_monthly['avg_temp_c'].round(1)
    
    # 2. Aggregate to annual data
    df_annual = df_daily.groupby('year').agg(
        annual_precip_mm=('prcp', 'sum'),
        annual_temp_c=('temp', 'mean')
    ).reset_index()
    
    df_annual['annual_precip_mm'] = df_annual['annual_precip_mm'].round(1)
    df_annual['annual_temp_c'] = df_annual['annual_temp_c'].round(1)
    
    # Save CSVs
    raw_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(raw_dir, exist_ok=True)
    
    monthly_path = os.path.join(raw_dir, "vancouver_climate_monthly_real.csv")
    annual_path = os.path.join(raw_dir, "vancouver_climate_annual_real.csv")
    
    df_monthly.to_csv(monthly_path, index=False)
    df_annual.to_csv(annual_path, index=False)
    
    print(f"Saved monthly climate data to {monthly_path}")
    print(f"Saved annual climate data to {annual_path}")
    return df_annual

def fetch_real_property_data(output_dir):
    print("Fetching property tax report from Vancouver Open Data Portal...")
    # Fetch 6000 records to ensure we get a solid sample after filtering
    url = 'https://opendata.vancouver.ca/api/records/1.0/search/?dataset=property-tax-report&rows=6000'
    r = requests.get(url)
    if r.status_code != 200:
        raise ConnectionError(f"Open Data API returned status code {r.status_code}")
        
    records = r.json().get('records', [])
    print(f"Retrieved {len(records)} raw property records.")
    
    # Extract fields into DataFrame
    df_properties = pd.DataFrame([rec['fields'] for rec in records])
    
    # Load neighbourhood mapping
    mapping_path = os.path.join(output_dir, "neighbourhood_mapping.json")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Static mapping file not found at {mapping_path}. Run build_neighborhood_mapping.py first.")
        
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        
    # Map BCAA neighborhood code to beach distance and neighborhood name
    def map_neighborhood(row):
        code = str(row.get('neighbourhood_code', ''))
        if code in mapping:
            return pd.Series([
                mapping[code]['neighbourhood_name'],
                mapping[code]['distance_to_beach_km']
            ])
        else:
            return pd.Series([None, None])
            
    df_properties[['neighbourhood_name', 'distance_to_beach_km']] = df_properties.apply(map_neighborhood, axis=1)
    
    raw_dir = os.path.join(output_dir, "raw_data")
    housing_path = os.path.join(raw_dir, "vancouver_housing_real.csv")
    df_properties.to_csv(housing_path, index=False)
    print(f"Saved housing records to {housing_path}")
    return df_properties

def merge_datasets(output_dir, df_properties, df_climate_annual):
    print("Merging housing and climate datasets with 1-year lag...")
    # Clean the housing DataFrame slightly for merging
    df_properties = df_properties.dropna(subset=['tax_assessment_year', 'neighbourhood_code', 'current_land_value'])
    df_properties['tax_assessment_year'] = df_properties['tax_assessment_year'].astype(int)
    
    # Create the merge key 'climate_year' (the year prior to the assessment year)
    df_properties['climate_year'] = df_properties['tax_assessment_year'] - 1
    
    # Merge on climate_year = year
    df_combined = pd.merge(
        df_properties, 
        df_climate_annual, 
        left_on='climate_year', 
        right_on='year', 
        how='left'
    )
    
    # Save combined raw data
    raw_dir = os.path.join(output_dir, "raw_data")
    combined_path = os.path.join(raw_dir, "vancouver_combined_real.csv")
    df_combined.to_csv(combined_path, index=False)
    print(f"Saved merged raw dataset to {combined_path} ({len(df_combined)} records)")
    return df_combined

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df_climate = fetch_real_climate_data(current_dir)
    df_properties = fetch_real_property_data(current_dir)
    merge_datasets(current_dir, df_properties, df_climate)
