import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from datetime import datetime

from model import CarPriceModel
from utils import export_prediction_pdf, export_prediction_csv, format_price_inr

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="🚗 India Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ───────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .price-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        margin: 1rem 0;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #333;
        border-left: 4px solid #667eea;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .sidebar .stSelectbox label, .sidebar .stSlider label {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Session State ───────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "df" not in st.session_state:
    st.session_state.df = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []

# ─────────────────────────── Header ───────────────────────────
st.markdown('<div class="main-title">🚗 India Used Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Random Forest · Upload your dataset to get started</div>', unsafe_allow_html=True)

# ─────────────────────────── Sidebar ───────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/car.png", width=80)
    st.markdown("## 📂 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ {len(df):,} records loaded!")
            st.markdown(f"**Columns:** {len(df.columns)}")
            st.markdown(f"**Shape:** `{df.shape}`")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")
    if st.session_state.df is not None:
        st.markdown("## ⚙️ Model Settings")
        n_estimators = st.slider("Number of Trees", 50, 500, 200, 50)
        max_depth = st.select_slider("Max Depth", options=[3, 5, 7, 10, 15, 20, None], value=10)
        test_size = st.slider("Test Split %", 10, 40, 20, 5)

        if st.button("🚀 Train Model"):
            with st.spinner("Training Random Forest..."):
                model = CarPriceModel(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    test_size=test_size / 100,
                )
                success, msg = model.train(st.session_state.df)
                if success:
                    st.session_state.model = model
                    st.session_state.trained = True
                    st.success("✅ Model trained!")
                else:
                    st.error(f"Training failed: {msg}")

    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "📊 EDA & Charts", "🔮 Price Prediction", "📈 Model Performance"],
        label_visibility="collapsed",
    )

# ─────────────────────────── Main Content ───────────────────────────
if st.session_state.df is None:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; background: #f8f9ff; border-radius: 20px; border: 2px dashed #667eea;">
            <div style="font-size: 4rem;">📁</div>
            <h3 style="color: #667eea;">Upload Your Dataset</h3>
            <p style="color: #666;">Upload a CSV file with India used car data from the sidebar to get started.</p>
            <p style="color: #888; font-size: 0.85rem;">Supported columns: Car Name, Brand, Year, Fuel Type, Transmission, KMs Driven, Price, etc.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Expected Column Format")
        sample = pd.DataFrame({
            "Car_Name": ["Maruti Swift", "Honda City"],
            "Year": [2019, 2020],
            "Selling_Price": [5.5, 9.0],
            "Present_Price": [8.0, 12.5],
            "Driven_kms": [35000, 22000],
            "Fuel_Type": ["Petrol", "Diesel"],
            "Selling_type": ["Dealer", "Individual"],
            "Transmission": ["Manual", "Automatic"],
            "Owner": [0, 1],
        })
        st.dataframe(sample, use_container_width=True)

else:
    df = st.session_state.df

    # ──────────── OVERVIEW ────────────
    if page == "🏠 Overview":
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Rows", df.duplicated().sum())

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**📄 Raw Data Preview**")
            st.dataframe(df.head(10), use_container_width=True)
        with col_right:
            st.markdown("**📊 Statistical Summary**")
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("**🔍 Column Info**")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(col_info, use_container_width=True)

    # ──────────── EDA ────────────
    elif page == "📊 EDA & Charts":
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

        model_obj = st.session_state.model
        target_col = model_obj.target_col if model_obj else None

        # Auto-detect price column
        price_candidates = [c for c in df.columns if any(k in c.lower() for k in ["price", "cost", "value", "sell"])]
        if not price_candidates:
            price_candidates = df.select_dtypes(include=np.number).columns.tolist()

        target_col = st.selectbox("Select Price (Target) Column", price_candidates, index=0)

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "🔗 Correlations", "🏷️ Category Analysis", "📅 Year Trends"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=target_col, nbins=50, title=f"Distribution of {target_col}",
                                   color_discrete_sequence=["#667eea"])
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, y=target_col, title=f"Box Plot – {target_col}",
                             color_discrete_sequence=["#764ba2"])
                st.plotly_chart(fig, use_container_width=True)

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) > 1:
                selected_num = st.multiselect("Select numeric columns for distribution", num_cols,
                                              default=num_cols[:min(4, len(num_cols))])
                if selected_num:
                    fig = make_subplots(rows=1, cols=len(selected_num), subplot_titles=selected_num)
                    colors = px.colors.qualitative.Set2
                    for i, col_name in enumerate(selected_num):
                        fig.add_trace(go.Histogram(x=df[col_name], name=col_name,
                                                   marker_color=colors[i % len(colors)]), row=1, col=i + 1)
                    fig.update_layout(height=300, showlegend=False, title="Numeric Column Distributions")
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            num_df = df.select_dtypes(include=np.number)
            if len(num_df.columns) >= 2:
                corr = num_df.corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                title="Correlation Heatmap", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Top Correlations with Target**")
                if target_col in corr.columns:
                    top_corr = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
                    fig2 = px.bar(top_corr, orientation="h", title=f"Feature Correlation with {target_col}",
                                  color=top_corr.values, color_continuous_scale="Viridis")
                    st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            if cat_cols:
                selected_cat = st.selectbox("Select Categorical Column", cat_cols)
                col1, col2 = st.columns(2)
                with col1:
                    vc = df[selected_cat].value_counts().reset_index()
                    vc.columns = [selected_cat, "Count"]
                    fig = px.bar(vc, x=selected_cat, y="Count", color=selected_cat,
                                 title=f"Count by {selected_cat}", color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if target_col in df.columns:
                        fig = px.box(df, x=selected_cat, y=target_col, color=selected_cat,
                                     title=f"{target_col} by {selected_cat}",
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns found in dataset.")

        with tab4:
            year_candidates = [c for c in df.columns if "year" in c.lower()]
            if year_candidates:
                year_col = year_candidates[0]
                # Ensure target column is numeric (handles currency strings like "₹ 1,95,000")
                df_plot = df.copy()
                df_plot[target_col] = (
                    df_plot[target_col].astype(str)
                    .str.replace(r"[₹,\s]", "", regex=True)
                    .pipe(pd.to_numeric, errors="coerce")
                )
                if df_plot[target_col].median() > 1000:
                    df_plot[target_col] = df_plot[target_col] / 1_00_000
                yearly = df_plot.groupby(year_col)[target_col].agg(["mean", "count"]).reset_index()
                yearly.columns = [year_col, "Avg Price", "Count"]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=yearly[year_col], y=yearly["Count"], name="Count",
                                     marker_color="#c3cfe2"), secondary_y=True)
                fig.add_trace(go.Scatter(x=yearly[year_col], y=yearly["Avg Price"], name="Avg Price",
                                         line=dict(color="#667eea", width=3), mode="lines+markers"), secondary_y=False)
                fig.update_layout(title="Avg Price & Volume by Year")
                st.plotly_chart(fig, use_container_width=True)

                car_age = datetime.now().year - df_plot[year_col]
                fig2 = px.scatter(df_plot, x=car_age, y=target_col, title="Car Age vs Selling Price",
                                  color_discrete_sequence=["#667eea"], opacity=0.5,
                                  labels={"x": "Car Age (Years)", "y": target_col})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No year column detected for trend analysis.")

    # ──────────── PREDICTION ────────────
    elif page == "🔮 Price Prediction":
        st.markdown('<div class="section-header">🔮 Predict Car Price</div>', unsafe_allow_html=True)

        if not st.session_state.trained:
            st.warning("⚠️ Please train the model first from the sidebar!")
        else:
            model_obj = st.session_state.model
            feature_info = model_obj.feature_info

            st.markdown("#### Fill in Car Details")
            input_data = {}
            cols = st.columns(3)
            col_idx = 0

            for feat, info in feature_info.items():
                with cols[col_idx % 3]:
                    if info["type"] == "categorical":
                        input_data[feat] = st.selectbox(f"**{feat}**", info["options"])
                    elif info["type"] == "numeric":
                        mn, mx, med = info["min"], info["max"], info["median"]
                        input_data[feat] = st.number_input(
                            f"**{feat}**", min_value=float(mn), max_value=float(mx),
                            value=float(med), step=float(max((mx - mn) / 100, 0.1))
                        )
                col_idx += 1

            st.markdown("---")
            col_btn, col_out = st.columns([1, 2])
            with col_btn:
                predict_btn = st.button("💡 Predict Price")

            if predict_btn:
                input_df = pd.DataFrame([input_data])
                price, low, high = model_obj.predict(input_df)

                with col_out:
                    st.markdown(f'<div class="price-box">💰 {format_price_inr(price)}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Price Range:** {format_price_inr(low)} – {format_price_inr(high)}")

                record = {**input_data, "Predicted_Price": round(price, 2),
                          "Lower_Bound": round(low, 2), "Upper_Bound": round(high, 2),
                          "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                st.session_state.predictions_history.append(record)

                # Feature importance mini-chart
                fi = model_obj.get_feature_importance()
                if fi is not None:
                    top_fi = fi.head(8)
                    fig = px.bar(top_fi, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Viridis",
                                 title="Top Feature Importances")
                    fig.update_layout(height=300, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

                # Export options
                st.markdown("#### 📥 Export Prediction")
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    csv_bytes = export_prediction_csv(st.session_state.predictions_history)
                    st.download_button("⬇️ Download CSV", csv_bytes, "predictions.csv",
                                       "text/csv", use_container_width=True)
                with exp_col2:
                    pdf_bytes = export_prediction_pdf(input_data, price, low, high)
                    st.download_button("⬇️ Download PDF Report", pdf_bytes, "prediction_report.pdf",
                                       "application/pdf", use_container_width=True)

            # Prediction History
            if st.session_state.predictions_history:
                st.markdown("#### 🕒 Prediction History")
                hist_df = pd.DataFrame(st.session_state.predictions_history)
                st.dataframe(hist_df, use_container_width=True)
                if st.button("🗑️ Clear History"):
                    st.session_state.predictions_history = []
                    st.rerun()

    # ──────────── MODEL PERFORMANCE ────────────
    elif page == "📈 Model Performance":
        st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)

        if not st.session_state.trained:
            st.warning("⚠️ Train the model first from the sidebar!")
        else:
            model_obj = st.session_state.model
            metrics = model_obj.metrics

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R² Score (Train)", f"{metrics['train_r2']:.4f}")
            col2.metric("R² Score (Test)", f"{metrics['test_r2']:.4f}")
            col3.metric("RMSE", f"{metrics['rmse']:.4f}")
            col4.metric("MAE", f"{metrics['mae']:.4f}")

            col_left, col_right = st.columns(2)
            with col_left:
                # Actual vs Predicted
                fig = px.scatter(x=model_obj.y_test, y=model_obj.y_pred,
                                 labels={"x": "Actual Price", "y": "Predicted Price"},
                                 title="Actual vs Predicted Price",
                                 color_discrete_sequence=["#667eea"], opacity=0.6)
                mn_val = min(model_obj.y_test.min(), model_obj.y_pred.min())
                mx_val = max(model_obj.y_test.max(), model_obj.y_pred.max())
                fig.add_trace(go.Scatter(x=[mn_val, mx_val], y=[mn_val, mx_val],
                                         mode="lines", name="Perfect Fit",
                                         line=dict(color="#764ba2", dash="dash")))
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Residual plot
                residuals = np.array(model_obj.y_test) - np.array(model_obj.y_pred)
                fig2 = px.scatter(x=model_obj.y_pred, y=residuals,
                                  labels={"x": "Predicted Price", "y": "Residuals"},
                                  title="Residual Plot", color_discrete_sequence=["#764ba2"], opacity=0.6)
                fig2.add_hline(y=0, line_dash="dash", line_color="#667eea")
                st.plotly_chart(fig2, use_container_width=True)

            # Feature Importance
            fi = model_obj.get_feature_importance()
            if fi is not None:
                fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                              color="Importance", color_continuous_scale="Viridis",
                              title="Feature Importances (All Features)")
                fig3.update_layout(yaxis=dict(autorange="reversed"), height=max(300, len(fi) * 28))
                st.plotly_chart(fig3, use_container_width=True)

            # Error distribution
            residuals_df = pd.DataFrame({"Residuals": residuals})
            fig4 = px.histogram(residuals_df, x="Residuals", nbins=40,
                                title="Error Distribution", color_discrete_sequence=["#667eea"])
            st.plotly_chart(fig4, use_container_width=True)