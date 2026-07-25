import os
import sys
import webbrowser
from datetime import datetime
from threading import Timer

import matplotlib
matplotlib.use('Agg')  # Suppress Matplotlib mainthread warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==============================================================================
# 0. AUTOMATIC BROWSER OPENER FOR LOCAL EXECUTION
# ==============================================================================
def open_browser():
    webbrowser.open_new_tab("http://localhost:8501")


if __name__ == "__main__":
    if not st.runtime.exists():
        from streamlit.web import cli as stcli

        Timer(2.0, open_browser).start()
        sys.argv = [
            "streamlit",
            "run",
            __file__,
            "--browser.gatherUsageStats=false",
            "--server.headless=true"
        ]
        sys.exit(stcli.main())

# ==============================================================================
# 1. CORE MACHINE LEARNING PIPELINE
# ==============================================================================
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

MODEL_DIR = "meditrak_model"
REGRESSOR_PATH = os.path.join(MODEL_DIR, "demand_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "demand_scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "category_encoder.pkl")

# File paths for datasets
SALES_CSV = "sales_transactions.csv"
CALENDAR_CSV = "calendar_dates.csv"
ITEMS_CSV = "pharmacy_products.csv"


def generate_fallback_datasets():
    """Generates realistic structured datasets if files are missing."""
    if not (os.path.exists(SALES_CSV) and os.path.exists(CALENDAR_CSV) and os.path.exists(ITEMS_CSV)):
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", end="2026-07-20", freq="D")
        stores = [101, 102, 103, 104]
        items = [2001, 2002, 2003, 2004, 2005, 2006]

        # 1. Sales Transactions
        sales_list = []
        for d in dates:
            for s in stores:
                for i in items:
                    base = 20 if i in [2001, 2002] else 12
                    wknd = 8 if d.weekday() >= 5 else 0
                    units = int(max(0, base + wknd + np.random.normal(0, 5)))
                    sales_list.append([d.strftime('%Y-%m-%d'), s, i, units])
        pd.DataFrame(sales_list, columns=['Date', 'StoreID', 'ItemID', 'UnitsSold']).to_csv(SALES_CSV, index=False)

        # 2. Calendar Dates and Seasons
        cal_list = []
        seasons = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                   6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall'}
        for d in dates:
            is_hol = 1 if (d.month == 12 and d.day == 25) or (d.month == 1 and d.day == 1) else 0
            is_prom = 1 if np.random.rand() < 0.1 else 0
            cal_list.append([d.strftime('%Y-%m-%d'), seasons[d.month], is_hol, is_prom])
        pd.DataFrame(cal_list, columns=['Date', 'Season', 'IsHoliday', 'IsPromotional']).to_csv(CALENDAR_CSV, index=False)

        # 3. Pharmacy Products Dataset
        products = [
            [2001, 'Pain Relief', 'Analgesics', 5.50],
            [2002, 'Infection Control', 'Antibiotics', 14.20],
            [2003, 'Allergy Care', 'Antihistamines', 7.80],
            [2004, 'Heart Health', 'Cardiovascular', 28.00],
            [2005, 'Diabetes Support', 'Diabetic Care', 22.10],
            [2006, 'Respiratory', 'Pulmonology', 16.40]
        ]
        pd.DataFrame(products, columns=['ItemID', 'ProductGroup', 'Category', 'BasePrice']).to_csv(ITEMS_CSV, index=False)


def run_ml_pipeline():
    """Ingests datasets, transforms features, and trains the model."""
    generate_fallback_datasets()

    # Ingest Datasets
    df_sales = pd.read_csv(SALES_CSV)
    df_calendar = pd.read_csv(CALENDAR_CSV)
    df_items = pd.read_csv(ITEMS_CSV)

    # Standardize column naming
    df_sales.columns = df_sales.columns.str.strip()
    df_calendar.columns = df_calendar.columns.str.strip()
    df_items.columns = df_items.columns.str.strip()

    # Merge Relational Tables
    df = pd.merge(df_sales, df_calendar, on='Date', how='left')
    df = pd.merge(df, df_items, on='ItemID', how='left')
    df = df.drop_duplicates()

    # Feature Engineering
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Categorical Encodings
    label_enc = LabelEncoder()
    df['Category_Encoded'] = label_enc.fit_transform(df['Category'].astype(str))

    # Target & Features
    feature_cols = [
        'StoreID', 'ItemID', 'BasePrice', 'IsHoliday',
        'IsPromotional', 'DayOfWeek', 'Month', 'IsWeekend', 'Category_Encoded'
    ]
    X = df[feature_cols]
    y = df['UnitsSold']

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training Model
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Metrics Evaluation
    preds = model.predict(X_test_scaled)
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    # Persist Artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, REGRESSOR_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_enc, ENCODER_PATH)

    return df, metrics, X_train.shape, X_test.shape


# Execute Pipeline
df_merged, model_metrics, train_size, test_size = run_ml_pipeline()
demand_model = joblib.load(REGRESSOR_PATH)
demand_scaler = joblib.load(SCALER_PATH)
cat_encoder = joblib.load(ENCODER_PATH)

# ==============================================================================
# 2. STREAMLIT APPLICATION DASHBOARD
# ==============================================================================
st.set_page_config(page_title="Meditrak Demand Forecasting", layout="wide", page_icon="💊")

# Custom Round Tablet Pill Logo Vector SVG
MEDITRAK_TABLET_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 80" width="260" height="70">
  <defs>
    <linearGradient id="tabletGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0E7490" />
      <stop offset="100%" stop-color="#06B6D4" />
    </linearGradient>
  </defs>
  <!-- Circular Tablet Pill Icon -->
  <g transform="translate(10, 8)">
    <circle cx="32" cy="32" r="28" fill="url(#tabletGrad)" />
    <!-- Center Score Line in Tablet -->
    <line x1="12" y1="32" x2="52" y2="32" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" />
    <!-- Medical Cross Notch Accent -->
    <path d="M 32 20 V 44" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.9" />
  </g>
  <!-- Brand Name -->
  <text x="80" y="46" font-family="'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="32" fill="#0F172A">Medi<tspan fill="#06B6D4">trak</tspan></text>
  <text x="82" y="62" font-family="'Segoe UI', Roboto, sans-serif" font-weight="600" font-size="10" fill="#64748B" letter-spacing="1.5">DEMAND FORECASTING</text>
</svg>
"""

# Render Sidebar Logo
with st.sidebar:
    st.image(MEDITRAK_TABLET_LOGO_SVG, use_container_width=False)
    st.markdown("---")
    st.markdown("### 💊 **Meditrak Platform**")
    st.caption("Multi-store pharmaceutical demand forecasting and automated replenishment order generator.")

# Main Page Header
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image(MEDITRAK_TABLET_LOGO_SVG, width=220)
with col_title:
    st.title("Meditrak Demand Forecasting Platform")
    st.caption("Pharmaceutical Supply Chain Analytics & Stock Planning System")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Demand Predictor",
    "📦 Next-Month Target Inventory",
    "📊 Sales Trend Analytics",
    "⚙️ Pipeline Splits & Source Datasets"
])

# TAB 1: Real-time Forecasting Feature
with tab1:
    st.header("Predict Specific Item Demand")
    st.write("Configure parameters to project stock target allocations:")

    available_stores = sorted(df_merged['StoreID'].unique().tolist())
    available_items = sorted(df_merged['ItemID'].unique().tolist())
    available_categories = sorted(df_merged['Category'].unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_store = st.selectbox("Select Pharmacy Branch ID", available_stores)
        selected_item = st.selectbox("Select Medicine Product ID", available_items)

        # Pull default price automatically from product catalog
        default_price = float(df_merged[df_merged['ItemID'] == selected_item]['BasePrice'].iloc[0])
        base_price = st.number_input("Base Unit Price ($)", min_value=1.0, max_value=200.0, value=default_price)

    with c2:
        target_month = st.slider("Target Projection Month", 1, 12, datetime.now().month)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        is_weekend = 1 if day_of_week >= 5 else 0
    with c3:
        is_holiday = st.checkbox("Is a Calendar Holiday?")
        is_promo = st.checkbox("Active Promotional Event?")
        category = st.selectbox("Medicine Category Context", available_categories)

    if st.button("Calculate Immediate Demand Target", type="primary"):
        cat_encoded = cat_encoder.transform([category])[0]
        input_data = [
            selected_store, selected_item, base_price,
            1 if is_holiday else 0, 1 if is_promo else 0,
            day_of_week, target_month, is_weekend, cat_encoded
        ]

        scaled_input = demand_scaler.transform([input_data])
        predicted_units = demand_model.predict(scaled_input)[0]

        st.markdown("---")
        st.subheader("💡 Forecast Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted Units Sold (Single Day)", f"{round(predicted_units, 2)} Units")
        res_col2.metric(
            "Recommended Replenishment Order Size",
            f"{int(predicted_units * 1.5)} Units",
            delta="Includes 50% Safety Buffer"
        )

# TAB 2: Strategic Supply Chain Order Generator
with tab2:
    st.header("📋 Next-Month Predicted Inventory Requirements")
    st.write("Automated replenishment recommendations based on historical patterns:")

    upcoming_records = []
    unique_items = df_merged[['ItemID', 'Category', 'BasePrice']].drop_duplicates()
    stores = df_merged['StoreID'].unique()

    for store in stores:
        for _, row in unique_items.iterrows():
            item_id = row['ItemID']
            cat_str = row['Category']
            price = row['BasePrice']
            cat_enc = cat_encoder.transform([cat_str])[0]

            mock_features = [store, item_id, price, 0, 0, 3, 8, 0, cat_enc]
            scaled = demand_scaler.transform([mock_features])
            monthly_estimate = int(demand_model.predict(scaled)[0] * 30)

            strategy = "Standard Replenishment" if monthly_estimate > 350 else "Low Volume (Reduce Risk)"

            upcoming_records.append({
                "Store ID": store,
                "Item ID": item_id,
                "Product Category": cat_str,
                "Base Cost Per Unit": f"${price:.2f}",
                "Estimated Next Month Demand (Units)": monthly_estimate,
                "Replenishment Guideline Action": strategy
            })

    st.dataframe(pd.DataFrame(upcoming_records), use_container_width=True)

# TAB 3: Line Charts and Volatility Spreads
with tab3:
    st.header("📈 Enterprise Sales Trend Metrics")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Seasonal Demand Trends (Monthly Volume)")
        fig1, ax1 = plt.subplots()
        monthly_trend = df_merged.groupby('Month')['UnitsSold'].sum().reset_index()
        sns.lineplot(data=monthly_trend, x='Month', y='UnitsSold', marker='o', color='#06B6D4', linewidth=2.5, ax=ax1)
        ax1.set_xticks(range(1, 13))
        st.pyplot(fig1)
        plt.close(fig1)

    with g2:
        st.subheader("Demand Volatility Across Store Branches")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df_merged, x='StoreID', y='UnitsSold', palette="mako", ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)

# TAB 4: Pipeline Architecture & Dataset Inspector
with tab4:
    st.header("⚙️ Pipeline Architecture & Raw Dataset Inspection")

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Merged Records", f"{df_merged.shape[0]} Rows")
    d2.metric("Training Set (80%)", f"{train_size[0]} Rows")
    d3.metric("Testing Set (20%)", f"{test_size[0]} Rows")

    st.markdown("---")
    st.subheader("🎯 Model Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score Accuracy", f"{model_metrics['R2']:.4f}")
    m2.metric("Mean Absolute Error (MAE)", f"{model_metrics['MAE']:.2f} Units")
    m3.metric("Mean Squared Error (MSE)", f"{round(model_metrics['MSE'], 2)}")

    st.markdown("---")
    st.subheader("🗃️ Ingested Source Datasets Preview")

    st.write(f"📁 **Sales Transactions Dataset (`{SALES_CSV}`)**")
    st.dataframe(pd.read_csv(SALES_CSV).head(5), use_container_width=True)

    st.write(f"📁 **Calendar Dates and Seasons (`{CALENDAR_CSV}`)**")
    st.dataframe(pd.read_csv(CALENDAR_CSV).head(5), use_container_width=True)

    st.write(f"📁 **Pharmacy Products Catalog (`{ITEMS_CSV}`)**")
    st.dataframe(pd.read_csv(ITEMS_CSV).head(5), use_container_width=True)