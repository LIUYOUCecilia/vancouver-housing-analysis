import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_vancouver_datasets(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Generating monthly climate dataset...")
    # Generate dates from Jan 2015 to Dec 2025 (11 years = 132 months)
    months = pd.date_range(start="2015-01-01", end="2025-12-01", freq="MS")
    climate_records = []
    
    for month in months:
        m_idx = month.month
        # Model precipitation: dry in summer (Jul-Aug: ~30-50mm), wet in winter (Nov-Jan: ~180-240mm)
        # Shift cosine wave so peak is in Dec (month 12) and trough is in Jun/Jul (month 6/7)
        base_precip = 140 + 90 * np.cos(2 * np.pi * (m_idx - 12) / 12)
        precip_noise = np.random.normal(0, 15)
        precipitation = max(0, float(base_precip + precip_noise))
        
        # Model temperature: peak in July/August
        base_temp = 11 + 9 * np.sin(2 * np.pi * (m_idx - 4) / 12)
        temp_noise = np.random.normal(0, 1.5)
        avg_temp = float(base_temp + temp_noise)
        
        climate_records.append({
            "year_month": month.strftime("%Y-%m"),
            "precipitation_mm": round(precipitation, 1),
            "avg_temp_c": round(avg_temp, 1)
        })
        
    df_climate = pd.DataFrame(climate_records)
    climate_path = os.path.join(raw_dir, "vancouver_climate.csv")
    df_climate.to_csv(climate_path, index=False)
    print(f"Climate dataset saved to {climate_path} ({len(df_climate)} records)")
    
    print("Generating housing transaction dataset...")
    num_transactions = 1200
    
    # Generate random transaction dates
    start_date = pd.to_datetime("2015-01-01")
    end_date = pd.to_datetime("2025-12-31")
    days_range = (end_date - start_date).days
    random_days = np.random.randint(0, days_range, size=num_transactions)
    dates = start_date + pd.to_timedelta(random_days, unit='D')
    
    # Generate bedrooms with realistic probabilities
    bedrooms_choices = [1, 2, 3, 4, 5]
    bedrooms_probs = [0.25, 0.40, 0.20, 0.10, 0.05]
    bedrooms = np.random.choice(bedrooms_choices, size=num_transactions, p=bedrooms_probs)
    
    # Generate distance to beach (0.1km to 15km)
    distance_to_beach = np.random.uniform(0.1, 15.0, size=num_transactions)
    
    # Generate property types and neighborhoods logically correlated with bedrooms
    property_types = []
    neighborhoods = []
    neighborhood_choices = ['Downtown', 'Kitsilano', 'West Point Grey', 'East Vancouver', 'Richmond', 'Burnaby']
    
    for beds in bedrooms:
        if beds <= 2:
            p_type = np.random.choice(['Condo', 'Townhouse'], p=[0.85, 0.15])
            neigh = np.random.choice(neighborhood_choices, p=[0.40, 0.20, 0.05, 0.15, 0.10, 0.10])
        elif beds == 3:
            p_type = np.random.choice(['Condo', 'Townhouse', 'Single-Family'], p=[0.20, 0.50, 0.30])
            neigh = np.random.choice(neighborhood_choices, p=[0.10, 0.25, 0.15, 0.20, 0.15, 0.15])
        else:
            p_type = np.random.choice(['Townhouse', 'Single-Family'], p=[0.10, 0.90])
            neigh = np.random.choice(neighborhood_choices, p=[0.02, 0.15, 0.35, 0.18, 0.15, 0.15])
            
        property_types.append(p_type)
        neighborhoods.append(neigh)
        
    # Calculate base price with statistical dependencies:
    # 1. Base price is 350k CAD
    # 2. Add 190k CAD per bedroom
    # 3. Subtract 65k CAD per km of distance to beach (beachfront premium)
    # 4. Long-term trend: Add 120 CAD per day since Jan 1, 2015 (market appreciation)
    # 5. Climate effect: Subtract 120 CAD per mm of precipitation in the transaction month
    #    (mimics a very small seasonal discount during heavy winter rains)
    # 6. Noise: Normal distribution with standard deviation of 45k CAD
    
    days_since_start = (dates - start_date).days.to_numpy()
    
    # Map climate precipitation to transactions
    temp_df = pd.DataFrame({"year_month": dates.strftime("%Y-%m")})
    temp_df = temp_df.merge(df_climate, on="year_month", how="left")
    precip_val = temp_df["precipitation_mm"].values
    
    prices = (
        380000 
        + 195000 * bedrooms 
        - 68000 * distance_to_beach 
        - 110 * precip_val 
        + 135 * days_since_start 
        + np.random.normal(0, 50000, size=num_transactions)
    )
    
    # Round prices to nearest hundred
    prices = np.round(prices / 100) * 100
    
    # Inject outliers to demonstrate the necessity of 1.5 * IQR data cleaning
    print("Injecting outliers...")
    outlier_indices = np.random.choice(range(num_transactions), size=12, replace=False)
    
    # 6 low-price outliers (e.g. data entry typos, missing a zero or foreclosure)
    for idx in outlier_indices[:6]:
        prices[idx] = np.round(prices[idx] / 10)  # Typo: e.g. 800k becomes 80k
        
    # 6 high-price outliers (extreme luxury estates not representative of standard market)
    for idx in outlier_indices[6:]:
        prices[idx] = prices[idx] * 8  # e.g. 1.2M becomes 9.6M
        
    df_housing = pd.DataFrame({
        "transaction_id": [f"TXN{i:04d}" for i in range(1, num_transactions + 1)],
        "date": dates.strftime("%Y-%m-%d"),
        "property_type": property_types,
        "bedrooms": bedrooms,
        "distance_to_beach_km": np.round(distance_to_beach, 2),
        "neighborhood": neighborhoods,
        "price_cad": prices.astype(float)
    })
    
    housing_path = os.path.join(raw_dir, "vancouver_housing.csv")
    df_housing.to_csv(housing_path, index=False)
    print(f"Housing transaction dataset saved to {housing_path} ({len(df_housing)} records)")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_vancouver_datasets(current_dir)
