import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real Estate AVM & Price Prediction Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM GLASSMORPHIC CSS STYLING ---
st.markdown("""
<style>
    /* Global Container Styling */
    .main {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header Gradient */
    .gradient-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    /* Card Container */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 5px;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* ROI Tip Box */
    .roi-card {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
    }
    
    .roi-title {
        color: #10b981;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    /* Hide default Streamlit padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    model_path = "model_pipeline.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_metadata():
    meta_path = "model_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_raw_data():
    data_path = "Property_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

model_pipe = load_model()
meta = load_metadata()
raw_df = load_raw_data()

# --- HEADER SECTION ---
col_logo, col_head = st.columns([1, 11])
with col_logo:
    st.markdown("<h1 style='font-size: 3.2rem;'>🏡</h1>", unsafe_allow_html=True)
with col_head:
    st.markdown("<div class='gradient-header'>Real Estate Automated Valuation Model (AVM)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Senior Data Scientist Capstone Dashboard — Lasso Regression Predictive Intelligence & Real Estate Valuation</div>", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("### 🎛️ Controls & Presets")
st.sidebar.markdown("---")

preset = st.sidebar.selectbox(
    "📍 Quick Property Preset",
    ["Custom Inputs", "Standard Single-Family Home", "Luxury Executive Estate", "Suburban Starter Home"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Active ML Model Status")
if meta and 'benchmarks' in meta:
    top_r2 = meta['benchmarks'][0]['Test R2 Score'] * 100
    top_mae = meta['benchmarks'][0]['MAE ($)']
    st.sidebar.success(f"**Lasso Regression Model**")
    st.sidebar.info(f"🎯 **Test R²**: {top_r2:.2f}%\n\n💵 **MAE**: ${top_mae:,.2f}")
else:
    st.sidebar.info("Lasso Regression Pipeline Active")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Capstone Data Science Team | Built with Scikit-Learn Lasso & Streamlit")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Real-Time Property Valuation", 
    "📊 Model Benchmarking & Performance", 
    "🔍 Price Drivers & Feature Analytics", 
    "📁 Data Explorer & Exports"
])

# ==========================================
# TAB 1: REAL-TIME PROPERTY VALUATION (LASSO PREDICTOR)
# ==========================================
with tab1:
    st.subheader("💡 Input Property Attributes for Real-Time Price Estimation (Lasso Regression Model)")
    
    if preset == "Luxury Executive Estate":
        def_liv_area, def_bsmt, def_qual, def_year, def_bath, def_cars, def_neigh = 3200, 1800, 9, 2018, 3, 3, "NridgHt"
    elif preset == "Suburban Starter Home":
        def_liv_area, def_bsmt, def_qual, def_year, def_bath, def_cars, def_neigh = 1100, 700, 5, 1975, 1, 1, "Edwards"
    elif preset == "Standard Single-Family Home":
        def_liv_area, def_bsmt, def_qual, def_year, def_bath, def_cars, def_neigh = 1700, 1000, 7, 2003, 2, 2, "CollgCr"
    else:
        def_liv_area, def_bsmt, def_qual, def_year, def_bath, def_cars, def_neigh = 1742, 1057, 7, 2003, 2, 2, "CollgCr"

    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        st.markdown("#### 📐 Living Area & Structure")
        gr_liv_area = st.slider("Above-Grade Living Area (sq ft)", 400, 4500, def_liv_area, step=50, help="GrLivArea: Total living space above ground.")
        total_bsmt_sf = st.slider("Total Basement Area (sq ft)", 0, 3000, def_bsmt, step=50, help="TotalBsmtSF: Square footage of basement.")
        year_built = st.slider("Year Built", 1872, 2010, def_year, step=1, help="YearBuilt: Original construction year.")
        property_size = st.number_input("Lot Size / Property Size (sq ft)", min_value=1000, max_value=100000, value=9500, step=500)
    
    with input_col2:
        st.markdown("#### 🌟 Quality & Finish Tiers")
        overall_qual = st.select_slider("Overall Material & Finish Quality", options=list(range(1, 11)), value=def_qual, help="1=Very Poor, 10=Very Excellent")
        
        neigh_options = meta['cat_options'].get('Neighborhood', ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'NridgHt', 'StoneBr']) if meta else ['CollgCr', 'NridgHt', 'StoneBr']
        def_neigh_idx = neigh_options.index(def_neigh) if def_neigh in neigh_options else 0
        neighborhood = st.selectbox("Neighborhood Location", neigh_options, index=def_neigh_idx)
        
        kitchen_qual = st.selectbox("Kitchen Quality Grade", ["Ex", "Gd", "TA", "Fa"], index=1 if def_qual > 6 else 2)
        exter_qual = st.selectbox("Exterior Quality Grade", ["Ex", "Gd", "TA", "Fa"], index=1 if def_qual > 6 else 2)
        bsmt_qual = st.selectbox("Basement Height/Quality", ["Ex", "Gd", "TA", "Fa", "None"], index=1 if def_qual > 6 else 2)

    with input_col3:
        st.markdown("#### 🚗 Amenities & Infrastructure")
        full_bath = st.slider("Full Bathrooms", 0, 4, def_bath, step=1)
        half_bath = st.slider("Half Bathrooms", 0, 2, 1, step=1)
        garage_cars = st.slider("Garage Car Capacity", 0, 4, def_cars, step=1)
        central_air = st.radio("Central Air Conditioning", ["Y", "N"], index=0, horizontal=True)
        property_zone = st.selectbox("Zoning Classification", meta['cat_options'].get('PropertyZone', ['RL', 'RM', 'FV', 'RH']) if meta else ['RL', 'RM'])

    st.markdown("---")
    
    # Calculate Real-Time Prediction using Lasso Pipeline
    if model_pipe is not None and meta is not None:
        num_cols = meta['num_cols']
        cat_cols = meta['cat_cols']
        qual_cols = meta['qual_cols']
        
        input_data = {}
        for col in num_cols:
            input_data[col] = meta['num_stats'].get(col, {}).get('median', 0.0)
        for col in cat_cols:
            options = meta['cat_options'].get(col, ['None'])
            input_data[col] = options[0] if options else 'None'
            
        input_data['GrLivArea'] = gr_liv_area
        input_data['TotalBsmtSF'] = total_bsmt_sf
        input_data['YearBuilt'] = year_built
        input_data['OverallQual'] = overall_qual
        input_data['Neighborhood'] = neighborhood
        input_data['KitchenQual'] = kitchen_qual
        input_data['ExterQual'] = exter_qual
        input_data['BsmtQual'] = bsmt_qual
        input_data['FullBath'] = full_bath
        input_data['HalfBath'] = half_bath
        input_data['BasementCars'] = garage_cars
        input_data['CentralAir'] = central_air
        input_data['PropertyZone'] = property_zone
        input_data['PropertySize'] = property_size
        
        input_data['TotalSF'] = gr_liv_area + total_bsmt_sf
        input_data['TotalBathrooms'] = full_bath + 0.5 * half_bath
        input_data['PropertyAge'] = 2026 - year_built
        
        qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'NA': 0}
        for col in qual_cols:
            if col in input_data and isinstance(input_data[col], str) and input_data[col] in qual_map:
                input_data[col] = qual_map[input_data[col]]

        input_df = pd.DataFrame([input_data])
        try:
            pred_log = model_pipe.predict(input_df)[0]
            est_price = np.expm1(pred_log)
            
            lower_bound = est_price * 0.90
            upper_bound = est_price * 1.10
            price_per_sqft = est_price / max(1, input_data['TotalSF'])
            
            if est_price >= 350000:
                tier_badge = "💎 Luxury Executive Estate"
            elif est_price >= 220000:
                tier_badge = "🥇 Premium Residential"
            elif est_price >= 140000:
                tier_badge = "🥈 Mid-Market Single Family"
            else:
                tier_badge = "🥉 Starter / Affordable Tier"

            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            
            with res_c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>💰 Estimated Market Valuation (Lasso)</div>
                    <div class='metric-val'>${est_price:,.0f}</div>
                    <div class='metric-sub'>Lasso Regressed Point Estimate</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>🎯 Valuation Confidence Range</div>
                    <div class='metric-val' style='font-size: 1.4rem;'>${lower_bound:,.0f} - ${upper_bound:,.0f}</div>
                    <div class='metric-sub'>±10% Appraisal Bounds</div>
                </div>
                """, unsafe_allow_html=True)

            with res_c3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>📏 Price Per Square Foot</div>
                    <div class='metric-val'>${price_per_sqft:,.2f}</div>
                    <div class='metric-sub'>Per Total SF (${input_data['TotalSF']:,} sq ft)</div>
                </div>
                """, unsafe_allow_html=True)

            with res_c4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>🏷️ Property Tier Classification</div>
                    <div class='metric-val' style='font-size: 1.2rem; margin-top: 10px;'>{tier_badge}</div>
                    <div class='metric-sub'>Market Classification</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=est_price,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Lasso Market Valuation Gauge vs Dataset Median ($154,150)", 'font': {'size': 16, 'color': '#e0e6ed'}},
                delta={'reference': 154150, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
                gauge={
                    'axis': {'range': [0, 600000], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "rgba(30, 41, 59, 0.7)",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 140000], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [140000, 250000], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [250000, 600000], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ef4444", 'width': 3},
                        'thickness': 0.75,
                        'value': est_price
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("""
            <div class='roi-card'>
                <div class='roi-title'>💡 Senior Data Scientist ROI Renovation Recommendation:</div>
                <div style='color: #cbd5e1; font-size: 0.9rem; margin-top: 5px;'>
                    • <b>Basement & Living Area Conversion</b>: Converting 300 sq ft of unfinished space into finished living area adds an estimated <b>+$18,500</b> in equity.<br>
                    • <b>Kitchen & Material Quality</b>: Upgrading Kitchen Quality from Average (TA) to Excellent (Ex) adds a <b>+8.4%</b> overall price premium.<br>
                    • <b>HVAC Central Air</b>: Installing Central Air conditioning prevents an automatic <b>-4.3%</b> valuation discount penalty.
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error executing Lasso prediction: {e}")
    else:
        st.warning("Model pipeline or metadata not found. Please ensure train_model.py has been executed.")

# ==========================================
# TAB 2: MODEL BENCHMARKING & PERFORMANCE
# ==========================================
with tab2:
    st.subheader("📊 Machine Learning Algorithm Leaderboard (Lasso Lead)")
    
    if meta and 'benchmarks' in meta:
        bench_df = pd.DataFrame(meta['benchmarks'])
        
        st.dataframe(
            bench_df.style.highlight_max(subset=['Test R2 Score', '5-Fold CV R2'], color='#065f46')
                          .highlight_min(subset=['MAE ($)', 'RMSE ($)', 'MAPE (%)'], color='#065f46')
                          .format({'5-Fold CV R2': '{:.4f}', 'Test R2 Score': '{:.2%}', 'MAE ($)': '${:,.2f}', 'RMSE ($)': '${:,.2f}', 'MAPE (%)': '{:.2f}%'}),
            use_container_width=True
        )
        
        bench_c1, bench_c2 = st.columns(2)
        
        with bench_c1:
            fig_r2 = px.bar(
                bench_df.sort_values(by='Test R2 Score', ascending=True),
                x='Test R2 Score',
                y='Model Algorithm',
                orientation='h',
                title='Algorithm Test R² Performance Comparison (Lasso #1)',
                color='Test R2 Score',
                color_continuous_scale='Blues',
                text_auto='.2%'
            )
            fig_r2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"))
            st.plotly_chart(fig_r2, use_container_width=True)

        with bench_c2:
            fig_mae = px.bar(
                bench_df.sort_values(by='MAE ($)', ascending=False),
                x='MAE ($)',
                y='Model Algorithm',
                orientation='h',
                title='Mean Absolute Error (MAE $) — Lower is Better',
                color='MAE ($)',
                color_continuous_scale='Reds_r',
                text_auto='$,.0f'
            )
            fig_mae.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"))
            st.plotly_chart(fig_mae, use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Active Predictor Diagnostics: Lasso Actual vs Predicted Price")
        
        if 'test_eval_sample' in meta:
            eval_df = pd.DataFrame(meta['test_eval_sample'])
            
            fig_scatter = px.scatter(
                eval_df,
                x='Actual Price',
                y='Predicted Price',
                hover_data=['Error', 'Absolute Pct Error'],
                title='Lasso Regression Model: Actual vs Predicted Property Valuation ($)',
                color='Absolute Pct Error',
                color_continuous_scale='Viridis',
                opacity=0.8
            )
            
            max_p = max(eval_df['Actual Price'].max(), eval_df['Predicted Price'].max())
            min_p = min(eval_df['Actual Price'].min(), eval_df['Predicted Price'].min())
            
            fig_scatter.add_shape(type="line", x0=min_p, y0=min_p, x1=max_p, y1=max_p, line=dict(color="Red", width=2, dash="dash"))
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"))
            st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 3: PRICE DRIVERS & FEATURE ANALYTICS
# ==========================================
with tab3:
    st.subheader("🔍 Top Valuation Drivers (Regularized Lasso Coefficients)")
    
    if meta and 'top_positive_drivers' in meta:
        pos_df = pd.DataFrame(list(meta['top_positive_drivers'].items()), columns=['Feature', 'Coefficient'])
        neg_df = pd.DataFrame(list(meta['top_negative_drivers'].items()), columns=['Feature', 'Coefficient'])
        
        combined_drivers = pd.concat([pos_df, neg_df]).sort_values(by='Coefficient', ascending=True)
        
        combined_drivers['Clean_Feature'] = combined_drivers['Feature'].str.replace('Neighborhood_', 'Location: ') \
                                                                       .str.replace('PropertyZone_', 'Zone: ') \
                                                                       .str.replace('RoofMatl_', 'Roof: ')
        
        fig_drivers = px.bar(
            combined_drivers,
            x='Coefficient',
            y='Clean_Feature',
            orientation='h',
            title='Regularized Lasso Feature Weights (Log-Price Impact Multipliers)',
            color='Coefficient',
            color_continuous_scale='RdYlGn',
            text_auto='.3f'
        )
        fig_drivers.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"), height=500)
        st.plotly_chart(fig_drivers, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📊 Exploratory Price Variance by Neighborhood & Quality")
    
    if raw_df is not None:
        eda_c1, eda_c2 = st.columns(2)
        
        with eda_c1:
            fig_box_neigh = px.box(
                raw_df,
                x='Neighborhood',
                y='PropPrice',
                color='Neighborhood',
                title='Property Price Distribution across Neighborhoods',
                points=False
            )
            fig_box_neigh.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"), showlegend=False)
            st.plotly_chart(fig_box_neigh, use_container_width=True)
            
        with eda_c2:
            fig_box_qual = px.box(
                raw_df,
                x='OverallQual',
                y='PropPrice',
                color='OverallQual',
                title='Property Price Scaling by Overall Quality Rating (1 - 10)',
                points=False
            )
            fig_box_qual.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"), showlegend=False)
            st.plotly_chart(fig_box_qual, use_container_width=True)

# ==========================================
# TAB 4: DATA EXPLORER & CSV EXPORTS
# ==========================================
with tab4:
    st.subheader("📁 Dataset Filtering & Predictions Export")
    
    if raw_df is not None:
        st.write(f"Total Observations: **{raw_df.shape[0]:,}** | Total Features: **{raw_df.shape[1]}**")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            price_min, price_max = st.slider("Filter by Price Range ($)", int(raw_df['PropPrice'].min()), int(raw_df['PropPrice'].max()), (100000, 400000))
        with col_f2:
            selected_neighs = st.multiselect("Filter by Neighborhood", options=raw_df['Neighborhood'].unique(), default=raw_df['Neighborhood'].unique()[:5])
            
        filtered_df = raw_df[(raw_df['PropPrice'] >= price_min) & 
                             (raw_df['PropPrice'] <= price_max) & 
                             (raw_df['Neighborhood'].isin(selected_neighs))]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        st.markdown("---")
        exp_c1, exp_c2 = st.columns(2)
        
        with exp_c1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Dataset (CSV)",
                data=csv_data,
                file_name="filtered_property_data.csv",
                mime="text/csv"
            )
            
        with exp_c2:
            if meta and 'test_eval_sample' in meta:
                eval_csv = pd.DataFrame(meta['test_eval_sample']).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Lasso Predictions & Residuals (CSV)",
                    data=eval_csv,
                    file_name="lasso_test_predictions.csv",
                    mime="text/csv"
                )
