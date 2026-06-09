import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

def validate_normal_equation(project_dir):
    data_path = os.path.join(project_dir, "processed_data", "vancouver_combined_cleaned.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError("Cleaned dataset not found! Run regression_analysis.py first.")
        
    df = pd.read_csv(data_path)
    
    # 1. Prepare data vectors
    # Y is the vector of target values (property values)
    Y = df["price_cad"].values
    
    # X is the matrix of features: [const, distance_to_beach_km, annual_precip_mm, age_at_assessment, is_strata]
    # We add a column of ones X_0 = 1 to represent the bias/intercept term
    N = len(df)
    ones = np.ones(N)
    features = df[["distance_to_beach_km", "annual_precip_mm", "age_at_assessment", "is_strata"]].values
    
    # Combine ones and features to create the full design matrix X
    X = np.column_stack((ones, features))
    
    print("Matrix dimensions for OLS Normal Equation:")
    print(f"  - Vector Y shape: {Y.shape} (Dimensions: {N} x 1)")
    print(f"  - Matrix X shape: {X.shape} (Dimensions: {N} x 5)")
    
    # 2. Compute Normal Equation step-by-step
    # Beta_hat = (X^T * X)^(-1) * X^T * Y
    print("\nComputing Normal Equation steps...")
    
    # Step 2.1: X^T * X
    XTX = X.T @ X
    print(f"  Step 1: Compute X^T @ X (Shape: {XTX.shape})")
    
    # Step 2.2: (X^T * X)^(-1)
    XTX_inv = np.linalg.inv(XTX)
    print(f"  Step 2: Compute Inverse of XTX (Shape: {XTX_inv.shape})")
    
    # Step 2.3: X^T * Y
    XTY = X.T @ Y
    print(f"  Step 3: Compute X^T @ Y (Shape: {XTY.shape})")
    
    # Step 2.4: Multiply inverse with XTY to find Beta_hat
    beta_hat = XTX_inv @ XTY
    print(f"  Step 4: Solve beta_hat = (XTX)^(-1) @ XTY")
    
    # 3. Fit benchmark OLS model via statsmodels for comparison
    X_sm = sm.add_constant(df[["distance_to_beach_km", "annual_precip_mm", "age_at_assessment", "is_strata"]])
    sm_model = sm.OLS(df["price_cad"], X_sm).fit()
    sm_coefficients = sm_model.params.values
    
    # 4. Print Comparison and Verify
    feature_names = [
        "Intercept (beta_0)", 
        "Beach Distance (beta_1)", 
        "Precipitation (beta_2)", 
        "Age (beta_3)",
        "Is Strata (beta_4)"
    ]
    
    print("\n" + "="*80)
    print("CS229 NORMAL EQUATION COEFFICIENTS VALIDATION (REAL DATA)")
    print("="*80)
    print(f"{'Feature':<25} | {'Normal Equation (NumPy)':<25} | {'Benchmark (Statsmodels)':<25}")
    print("-"*80)
    for name, b_hat, b_sm in zip(feature_names, beta_hat, sm_coefficients):
        print(f"{name:<25} | {b_hat:<25,.5f} | {b_sm:<25,.5f}")
    print("="*80)
    
    # Assert check
    try:
        np.testing.assert_allclose(beta_hat, sm_coefficients, rtol=1e-5, atol=1e-5)
        print("\n✅ SUCCESS: NumPy Normal Equation calculation matches Statsmodels benchmark perfectly!")
    except AssertionError as e:
        print("\n❌ FAILURE: Coefficients do not match. Check calculations.")
        print(e)
        
    return beta_hat

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    validate_normal_equation(project_dir)
