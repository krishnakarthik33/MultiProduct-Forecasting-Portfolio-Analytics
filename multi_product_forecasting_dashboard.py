# ==========================================
# MULTI-PRODUCT FORECASTING & PORTFOLIO ANALYTICS
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="Multi-Product Forecasting & Portfolio Analytics",
    layout="wide"
)

st.title("📦 Multi-Product Forecasting & Portfolio Analytics Dashboard")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Sales Excel / CSV File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df['date'] = pd.to_datetime(df['date'])
    df['revenue'] = df['demand'] * df['price']

    st.success("✅ Data loaded successfully")

    # -------------------------------
    # PRODUCT SELECTION
    # -------------------------------
    st.sidebar.header("🔍 Controls")

    products = st.sidebar.multiselect(
        "Select Products",
        df['product'].unique(),
        default=df['product'].unique()
    )

    filtered_df = df[df['product'].isin(products)]

    # -------------------------------
    # PORTFOLIO METRICS
    # -------------------------------
    st.subheader("📊 Portfolio Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Revenue",
        f"₹ {round(filtered_df['revenue'].sum(), 2)}"
    )

    col2.metric(
        "Total Demand",
        int(filtered_df['demand'].sum())
    )

    col3.metric(
        "Products Analyzed",
        filtered_df['product'].nunique()
    )

    # -------------------------------
    # REVENUE CONTRIBUTION
    # -------------------------------
    st.subheader("💰 Revenue Contribution by Product")

    revenue_by_product = (
        filtered_df
        .groupby("product")['revenue']
        .sum()
        .sort_values(ascending=False)
    )

    fig1, ax1 = plt.subplots()
    revenue_by_product.plot(kind='bar', ax=ax1)
    ax1.set_ylabel("Revenue")
    st.pyplot(fig1)

    # -------------------------------
    # DEMAND VOLATILITY ANALYSIS
    # -------------------------------
    st.subheader("📉 Demand Volatility (Risk Indicator)")

    volatility = (
        filtered_df
        .groupby("product")['demand']
        .std()
        .sort_values(ascending=False)
    )

    fig2, ax2 = plt.subplots()
    volatility.plot(kind='bar', ax=ax2)
    ax2.set_ylabel("Demand Std Dev")
    st.pyplot(fig2)

    # -------------------------------
    # FORECASTING SECTION
    # -------------------------------
    st.subheader("🔮 Product-Level Demand Forecasting")

    selected_product = st.selectbox(
        "Choose Product to Forecast",
        products
    )

    forecast_days = st.slider(
        "Forecast Days",
        7, 90, 30
    )

    product_df = filtered_df[filtered_df['product'] == selected_product]

    prophet_df = product_df[['date', 'demand']]
    prophet_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    fig3 = model.plot(forecast)
    st.pyplot(fig3)

    # -------------------------------
    # PRODUCT PERFORMANCE CLASSIFICATION
    # -------------------------------
    st.subheader("🏷️ Product Portfolio Classification")

    portfolio = (
        filtered_df
        .groupby("product")
        .agg(
            avg_demand=('demand', 'mean'),
            demand_volatility=('demand', 'std'),
            revenue=('revenue', 'sum')
        )
    )

    portfolio['category'] = np.where(
        (portfolio['revenue'] > portfolio['revenue'].median()) &
        (portfolio['demand_volatility'] < portfolio['demand_volatility'].median()),
        "⭐ Star Product",
        "⚠️ Risk / Low Performer"
    )

    st.dataframe(portfolio)

else:
    st.info("⬆️ Upload a sales data file to begin")
