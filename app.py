import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import os

# Set page config for a premium wide-screen look
st.set_page_config(
    page_title="Vancouver Real Estate & Climate Regression Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism, nice fonts, and premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    /* Compress default Streamlit margins and padding */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 0.8rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px -5px rgba(59, 130, 246, 0.4);
        margin-bottom: 0.6rem;
    }
    
    .prediction-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin-top: 0.2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .coefficient-card {
        background-color: #f8fafc;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .coefficient-title {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .coefficient-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.1rem;
    }
    
    .diagnostic-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 0.8rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data and fit model (cached)
@st.cache_data
def load_and_model_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "processed_data", "vancouver_combined_cleaned.csv")
    if not os.path.exists(data_path):
        # Fallback if cleaning hasn't run
        from data_cleaning import clean_and_merge_data
        df, _ = clean_and_merge_data(current_dir)
    else:
        df = pd.read_csv(data_path)
        
    # Fit OLS
    Y = df["price_cad"]
    X = df[["distance_to_beach_km", "annual_precip_mm", "age_at_assessment", "is_strata"]]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const).fit()
    
    return df, model

try:
    df, model = load_and_model_data()
    coef = model.params
    # Add consumer-friendly columns for visualization and legends
    df['Property Type'] = df['is_strata'].map({1: "Condo/Townhouse", 0: "Single-Family Home"})
    df['Age Group'] = pd.cut(
        df['age_at_assessment'],
        bins=[-1, 10, 30, 60, 999],
        labels=['New (< 10 Years)', 'Modern (10-30 Years)', 'Established (30-60 Years)', 'Historic (60+ Years)']
    )
except Exception as e:
    st.error(f"Error loading model data: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-title">Vancouver Real Estate & Climate Regression Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive portfolio showcase analyzing how location, building characteristics, and annual climate trends shape property valuations.</div>', unsafe_allow_html=True)

# Sidebar for User Inputs (grouped with the pricing prediction)
prediction_placeholder = st.sidebar.empty()

st.sidebar.markdown("###  Property Predictor Sliders")
st.sidebar.write("Adjust property attributes below to estimate its assessed value dynamically using OLS coefficients:")

# Set up sliders based on real dataset ranges
dist_beach = st.sidebar.slider(
    " Distance to Beach (km)", 
    min_value=float(df["distance_to_beach_km"].min()), 
    max_value=float(df["distance_to_beach_km"].max()), 
    value=2.0, 
    step=0.1
)

prop_age = st.sidebar.slider(
    "Property Age (Years)", 
    min_value=int(df["age_at_assessment"].min()), 
    max_value=int(df["age_at_assessment"].max()), 
    value=15, 
    step=1
)

prop_type = st.sidebar.selectbox(
    "Property Type",
    options=["Condo / Townhouse", "Single-Family Home"]
)
is_strata = 1 if "Condo" in prop_type else 0

ann_precip = st.sidebar.slider(
    " Annual Precipitation (mm)", 
    min_value=int(df["annual_precip_mm"].min()), 
    max_value=int(df["annual_precip_mm"].max()), 
    value=1400, 
    step=10
)

# Calculate prediction dynamically based on slider values
pred_val = (
    coef["const"] 
    + coef["distance_to_beach_km"] * dist_beach 
    + coef["annual_precip_mm"] * ann_precip 
    + coef["age_at_assessment"] * prop_age 
    + coef["is_strata"] * is_strata
)

# Update the placeholder at the top of the sidebar with the prediction card
prediction_placeholder.markdown(f"""
<div class="prediction-card">
    <div style="font-size: 0.85rem; opacity: 0.9;">Estimated Property Assessed Value</div>
    <div class="prediction-value" style="font-size: 2.2rem;">${pred_val:,.0f} CAD</div>
    <div style="margin-top: 5px; font-size: 0.7rem; opacity: 0.85;">
        *Computed using real-world OLS coefficients on 4,300+ Vancouver properties.
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model R-squared:** `{:.2f}%`
""".format(model.rsquared * 100))


# The chart filters have been moved directly inside the tabs to keep the sidebar clean.
df_filtered = df.copy()

# Layout: 2 Columns (Left: Coefficients, Right: Charts)
col_left, col_right = st.columns([1, 1.5])

with col_left:
    # 2. Coefficients List (Sleek cards)
    st.markdown("### 📊 Valuation Breakdown (Current Property)")
    st.write("Dynamic impact of each property attribute on the estimated valuation:")
    
    # Calculate individual components
    val_const = coef['const']
    val_beach = coef['distance_to_beach_km'] * dist_beach
    val_age = coef['age_at_assessment'] * prop_age
    val_strata = coef['is_strata'] * is_strata
    val_precip = coef['annual_precip_mm'] * ann_precip
    
    # 1. Base Value
    st.markdown(f"""
    <div class="coefficient-card">
        <div class="coefficient-title">Base Baseline Value</div>
        <div class="coefficient-value" style="color: #10b981;">+${val_const:,.0f} CAD</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;">
            Baseline market starting value for properties in Vancouver.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Beach Proximity
    sign_beach = "+" if val_beach >= 0 else "-"
    color_beach = "#10b981" if val_beach >= 0 else "#0f172a"
    st.markdown(f"""
    <div class="coefficient-card">
        <div class="coefficient-title">Beach Proximity Impact ({dist_beach:.1f} km)</div>
        <div class="coefficient-value" style="color: {color_beach};">{sign_beach}${abs(val_beach):,.0f} CAD</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;">
            Adjusted at -${abs(coef['distance_to_beach_km']):,.0f} CAD per km away from beach.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Building Age
    sign_age = "+" if val_age >= 0 else "-"
    color_age = "#10b981" if val_age >= 0 else "#0f172a"
    st.markdown(f"""
    <div class="coefficient-card">
        <div class="coefficient-title">Building Age Depreciation ({prop_age} Years)</div>
        <div class="coefficient-value" style="color: {color_age};">{sign_age}${abs(val_age):,.0f} CAD</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;">
            Depreciated at -${abs(coef['age_at_assessment']):,.0f} CAD per year of building age.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4. Property Type
    sign_strata = "+" if val_strata >= 0 else "-"
    color_strata = "#10b981" if val_strata >= 0 else "#0f172a"
    st.markdown(f"""
    <div class="coefficient-card">
        <div class="coefficient-title">Property Type Adjustment ({prop_type})</div>
        <div class="coefficient-value" style="color: {color_strata};">{sign_strata}${abs(val_strata):,.0f} CAD</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;">
            Average flat adjustment of -${abs(coef['is_strata']):,.0f} CAD for Condo/Townhouse units.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 5. Climate / Precipitation
    sign_precip = "+" if val_precip >= 0 else "-"
    color_precip = "#10b981" if val_precip >= 0 else "#0f172a"
    st.markdown(f"""
    <div class="coefficient-card">
        <div class="coefficient-title">Precipitation Impact ({ann_precip} mm)</div>
        <div class="coefficient-value" style="color: {color_precip};">{sign_precip}${abs(val_precip):,.0f} CAD</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;">
            Correlated at -${abs(coef['annual_precip_mm']):,.2f} CAD per mm of annual precipitation.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    # 3. Interactive Charts Tab
    tab1, tab2, tab3 = st.tabs([" Price vs. Beach Proximity", " 3D Feature Space Plane", " Model Diagnostics"])
    
    with tab1:
        st.markdown("#### Assessed Valuation vs. Distance to Beach")
        
        # Slider on top of the chart inside Tab 1
        age_range_tab1 = st.slider(
            "Adjust Displayed Property Age Range (Years):",
            min_value=int(df["age_at_assessment"].min()),
            max_value=int(df["age_at_assessment"].max()),
            value=(10, 50), # Default to a subset so we don't show all ages at once
            step=1,
            key="tab1_age_slider"
        )
        
        st.write("Scatter plot of properties colored by age. The right vertical bar represents the Property Age scale (0-120). Adjust the slider above to filter the properties displayed.")
        
        df_filtered_tab1 = df[(df["age_at_assessment"] >= age_range_tab1[0]) & (df["age_at_assessment"] <= age_range_tab1[1])]
        
        # Calculate predicted OLS line values
        mean_precip = df["annual_precip_mm"].mean()
        mean_age = df["age_at_assessment"].mean()
        mean_strata = df["is_strata"].mean()
        
        x_line = np.linspace(df["distance_to_beach_km"].min(), df["distance_to_beach_km"].max(), 100)
        y_line = (
            coef["const"] 
            + coef["distance_to_beach_km"] * x_line 
            + coef["annual_precip_mm"] * mean_precip 
            + coef["age_at_assessment"] * mean_age 
            + coef["is_strata"] * mean_strata
        )
        
        fig1 = px.scatter(
            df_filtered_tab1,
            x="distance_to_beach_km",
            y="price_cad",
            color="age_at_assessment",
            color_continuous_scale="viridis_r",
            range_color=[int(df["age_at_assessment"].min()), int(df["age_at_assessment"].max())],
            labels={
                "distance_to_beach_km": "Distance to Beach (km)",
                "price_cad": "Assessed Value (CAD)",
                "age_at_assessment": "Property Age (Years)"
            },
            hover_data=["neighbourhood_name", "tax_assessment_year"],
            opacity=0.5
        )
        
        # Add the regression line
        fig1.add_trace(go.Scatter(
            x=x_line, 
            y=y_line, 
            mode='lines', 
            name='OLS Fitted Regression', 
            line=dict(color='#dc2626', width=3)
        ))
        
        fig1.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(
                title="Property Age (Years)",
                thicknessmode="pixels", thickness=15,
                lenmode="fraction", len=0.8,
                yanchor="middle", y=0.5
            )
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab2:
        st.markdown("#### 3D View of Beach Proximity, Age, and Property Value")
        
        # Interactive filters directly inside Tab 2
        selected_types_tab2 = st.multiselect(
            "Select Property Types to Display:",
            options=["Condo/Townhouse", "Single-Family Home"],
            default=["Condo/Townhouse", "Single-Family Home"],
            key="tab2_types_filter"
        )
        
        age_range_tab2 = st.slider(
            "Adjust Displayed Property Age Range for 3D View (Years):",
            min_value=int(df["age_at_assessment"].min()),
            max_value=int(df["age_at_assessment"].max()),
            value=(10, 50), # Default to a subset
            step=1,
            key="tab2_age_slider"
        )
        
        st.write("A 3D scatter plot visualizing property values. The legend on the right shows the Property Type colors. Click on them to toggle categories on and off.")
        
        df_filtered_tab2 = df[
            df['Property Type'].isin(selected_types_tab2) &
            (df['age_at_assessment'] >= age_range_tab2[0]) &
            (df['age_at_assessment'] <= age_range_tab2[1])
        ]
        
        # Subsample to keep 3D plot highly responsive
        df_sub = df_filtered_tab2.sample(min(1500, len(df_filtered_tab2)), random_state=42) if len(df_filtered_tab2) > 0 else df_filtered_tab2
        
        if not df_sub.empty:
            fig2 = px.scatter_3d(
                df_sub,
                x="distance_to_beach_km",
                y="age_at_assessment",
                z="price_cad",
                color="Property Type",
                color_discrete_map={"Condo/Townhouse": "#10b981", "Single-Family Home": "#f43f5e"},
                labels={
                    "distance_to_beach_km": "Beach Dist (km)",
                    "age_at_assessment": "Building Age (Years)",
                    "price_cad": "Valuation (CAD)",
                    "Property Type": "Property Type"
                },
                hover_data=["neighbourhood_name"],
                opacity=0.7
            )
            
            fig2.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=0, b=0),
                scene=dict(
                    xaxis_title='Beach Distance (km)',
                    yaxis_title='Building Age (years)',
                    zaxis_title='Assessed Value (CAD)'
                ),
                legend=dict(
                    title="Property Type",
                    yanchor="top", y=0.99, xanchor="right", x=0.99
                )
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No data matches the selected filters. Adjust the controls above.")
        
    with tab3:
        st.markdown("#### Academic Diagnostic Summary & Residual Checks")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("""
            <div class="diagnostic-card">
                <h5 style="color: #0f172a; margin-top:0;"> Multicollinearity (VIF)</h5>
                <p style="font-size: 0.85rem; color: #64748b;">
                    VIF measures how much a feature's coefficient variance is inflated by collinearity. 
                    A VIF &gt; 10 represents high collinearity.
                </p>
                <table style="width:100%; font-size: 0.9rem; text-align:left; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #e2e8f0; color: #64748b;"><th style="padding: 4px;">Feature</th><th>VIF</th></tr>
                    <tr><td style="padding: 4px;">Distance to Beach</td><td><b>3.88</b></td></tr>
                    <tr style="color: #e11d48;"><td style="padding: 4px;">Annual Precipitation</td><td><b>16.03</b> (Collinear)</td></tr>
                    <tr><td style="padding: 4px;">Building Age</td><td><b>3.98</b></td></tr>
                    <tr><td style="padding: 4px;">Is Strata Property</td><td><b>4.92</b></td></tr>
                </table>
                <p style="font-size: 0.75rem; color: #64748b; margin-top: 10px; line-height: 1.2;">
                    *Note: The high VIF for annual precipitation arises from merging a year-level macro variable with property-level micro data, causing low variation within same-year cohorts.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with diag_col2:
            st.markdown("""
            <div class="diagnostic-card">
                <h5 style="color: #0f172a; margin-top:0;"> Heteroscedasticity (Breusch-Pagan)</h5>
                <p style="font-size: 0.85rem; color: #64748b;">
                    Breusch-Pagan tests if model residuals have a constant variance (homoscedasticity).
                </p>
                <table style="width:100%; font-size: 0.9rem; text-align:left; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #e2e8f0; color: #64748b;"><th style="padding: 4px;">Metric</th><th>Value</th></tr>
                    <tr><td style="padding: 4px;">LM Statistic</td><td><b>160.09</b></td></tr>
                    <tr><td style="padding: 4px;">LM-Test p-value</td><td><b>0.0000</b></td></tr>
                    <tr><td style="padding: 4px;">F-Statistic</td><td><b>41.51</b></td></tr>
                    <tr><td style="padding: 4px;">F-Test p-value</td><td><b>0.0000</b></td></tr>
                </table>
                <p style="font-size: 0.75rem; color: #e11d48; margin-top: 10px; line-height: 1.2; font-weight: 600;">
                    Conclusion: Reject Homoscedasticity (p < 0.05). Standard errors should be corrected using Robust Covariance matrices (HC1/HC3) in academic publications.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Add residual analysis plot
        df_res = pd.DataFrame({
            "Fitted": model.fittedvalues / 1e6,
            "Residuals": model.resid / 1e3
        })
        fig3 = px.scatter(
            df_res,
            x="Fitted",
            y="Residuals",
            opacity=0.4,
            labels={"Fitted": "Fitted Values (Millions CAD)", "Residuals": "Residuals (Thousands CAD)"},
            color_discrete_sequence=["#5b21b6"]
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="#ef4444", line_width=2)
        fig3.update_layout(
            height=300,
            title="Residuals vs. Fitted Values (Heteroscedasticity Visual Check)",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig3, use_container_width=True)
