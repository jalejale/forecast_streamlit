"""
app.py — Python Forecasting App (Streamlit)
Template: date, brand, sub_brand, qty

Run: streamlit run app.py
Opens: http://localhost:8501
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import forecasting as fc

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_COLS    = ["date", "brand", "sub_brand", "qty"]
SAMPLE_DATA_PATH = "sample_data.csv"

COLORS = {
    "actual":   "#a5b4fc",
    "fitted":   "#f59e0b",
    "forecast": "#38ef7d",
    "ci":       "rgba(56,239,125,0.15)",
    "trend":    "#f64f59",
    "seasonal": "#667eea",
    "resid":    "#fb923c",
}

PALETTE = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

FREQ_MAP = {
    "Monthly (MS)":   "MS",
    "Daily (D)":      "D",
    "Weekly (W)":     "W",
    "Quarterly (QS)": "QS",
}

METRIC_HINTS = {
    "MAE": "Mean Absolute Error: The average absolute difference between forecast and actuals. Lower is better.",
    "RMSE": "Root Mean Squared Error: The square root of average squared errors. Penalizes larger errors more than MAE. Lower is better.",
    "MAPE": "Mean Absolute Percentage Error: The average percentage difference between forecast and actuals. Lower is better.",
    "OBSERVATIONS": "Total number of recorded data points in this historical series.",
    "MIN QTY": "The lowest recorded quantity in the historical data.",
    "MAX QTY": "The highest recorded quantity in the historical data.",
    "MEAN QTY": "The average quantity over the historical data.",
    "TREND STRENGTH": "Measures how much variance is explained by the trend (0 to 1). Higher = stronger trend.",
    "SEASONAL STRENGTH": "Measures how much variance is explained by seasonality (0 to 1). Higher = stronger seasonality.",
    "RESIDUAL STD": "Standard deviation of the residuals (noise/error). Lower means the model captures more signal."
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def blank_template_bytes() -> bytes:
    buf = io.StringIO()
    buf.write(",".join(TEMPLATE_COLS) + "\n")
    buf.write("2024-01-01,BrandA,SubBrand-1,100\n")
    buf.write("2024-02-01,BrandA,SubBrand-1,110\n")
    return buf.getvalue().encode("utf-8")

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=True, sheet_name="Forecast")
    return buf.getvalue()

def parse_uploaded_st(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format.")
    
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in TEMPLATE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df["qty"]  = pd.to_numeric(df["qty"], errors="coerce")
    return df

@st.cache_data
def load_sample() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_series(df: pd.DataFrame, brand: str, sub_brand: str, freq: str) -> pd.Series:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["brand"] == brand) & (df["sub_brand"] == sub_brand)
    filtered = df[mask].copy().sort_values("date").set_index("date")["qty"]
    filtered.index = pd.DatetimeIndex(filtered.index)
    return filtered.asfreq(freq).interpolate()

def forecast_figure(series, result, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, name="Actual", mode="lines+markers",
        line=dict(color=COLORS["actual"], width=2), marker=dict(size=5),
    ))
    fitted = result.get("fitted")
    if fitted is not None:
        fig.add_trace(go.Scatter(
            x=fitted.index, y=fitted.values, name="Fitted", mode="lines",
            line=dict(color=COLORS["fitted"], width=1.5, dash="dot"),
        ))
    lower, upper, fc_val = result.get("lower"), result.get("upper"), result.get("forecast")
    if lower is not None and upper is not None and fc_val is not None:
        fig.add_trace(go.Scatter(
            x=list(fc_val.index) + list(fc_val.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor=COLORS["ci"],
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))
        fig.add_trace(go.Scatter(
            x=fc_val.index, y=fc_val.values, name="Forecast", mode="lines+markers",
            line=dict(color=COLORS["forecast"], width=2.5),
            marker=dict(size=6, symbol="diamond"),
        ))
    fig.add_vline(x=series.index[-1], line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
    
    # Generic streamit dark layout mappings
    fig.update_layout(
        title=title, hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", font=dict(color="#e8eaf6"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def display_metrics(metrics: dict):
    if not metrics:
        return
    cols = st.columns(min(len(metrics), 4) or 1)
    for i, (k, v) in enumerate(metrics.items()):
        with cols[i % 4]:
            st.metric(k, str(v), help=METRIC_HINTS.get(k, ""))

def render_forecast_output(series, result, label, periods, filename):
    st.divider()
    display_metrics(result.get("metrics", {}))
    st.plotly_chart(forecast_figure(series, result, f"{label} — {periods}-Period Forecast"), use_container_width=True)
    
    fc_df = pd.DataFrame({
        "Forecast":  result["forecast"],
        "Lower 95%": result.get("lower"),
        "Upper 95%": result.get("upper"),
    }).reset_index()
    fc_df.columns = ["Date", "Forecast", "Lower 95%", "Upper 95%"]
    fc_df["Date"] = fc_df["Date"].astype(str)
    
    with st.expander("📋 Forecast Table"):
        st.dataframe(fc_df.round(2), use_container_width=True, hide_index=True)
        
    st.download_button("⬇️ Download Forecast (Excel)", data=to_excel_bytes(fc_df), file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

def overview_tab(df, series, brand, sub_brand):
    n   = len(series.dropna())
    mn  = series.min()
    mx  = series.max()
    avg = series.mean()
    
    display_metrics({
        "OBSERVATIONS": n,
        "MIN QTY": f"{mn:,.0f}",
        "MAX QTY": f"{mx:,.0f}",
        "MEAN QTY": f"{avg:,.1f}"
    })
    
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers",
                              line=dict(color=COLORS["actual"], width=2), marker=dict(size=6), name="Qty"))
    fig0.update_layout(title=f"Qty Over Time — {brand} / {sub_brand}", hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
    )
    st.plotly_chart(fig0, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Descriptive Statistics")
        stats = series.describe().reset_index()
        stats.columns = ["Stat", "Qty"]
        st.dataframe(stats, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Raw Data (last 24)")
        raw24 = series.tail(24).reset_index()
        raw24.columns = ["Date", "Qty"]
        raw24["Date"] = raw24["Date"].astype(str)
        st.dataframe(raw24, use_container_width=True, hide_index=True)
        
    sub_brands = sorted(df.loc[df["brand"] == brand, "sub_brand"].dropna().unique().tolist())
    fig_cmp = go.Figure()
    for i, sb in enumerate(sub_brands):
        m_ = (df["brand"] == brand) & (df["sub_brand"] == sb)
        s_ = df[m_].sort_values("date").set_index("date")["qty"]
        fig_cmp.add_trace(go.Scatter(x=s_.index, y=s_.values, name=sb, mode="lines",
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    fig_cmp.update_layout(title=f"All Sub-Brands of '{brand}'", hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


def ma_tab(series):
    st.subheader("Moving Average")
    col1, col2 = st.columns(2)
    with col1:
        periods = st.number_input("Forecast Periods", value=12, min_value=1, max_value=36, key="ma_p")
    with col2:
        window = st.number_input("Window Size", value=3, min_value=2, max_value=24, key="ma_w")
    
    try:
        result = fc.moving_average(series, periods=int(periods), window=int(window))
        render_forecast_output(series, result, "Moving Average", int(periods), "ma_forecast.xlsx")
    except Exception as e:
        st.error(f"Error: {e}")

def ses_tab(series):
    st.subheader("Simple Exponential Smoothing (SES)")
    col1, col2 = st.columns(2)
    with col1:
        periods = st.number_input("Forecast Periods", value=12, min_value=1, max_value=36, key="ses_p")
    with col2:
        alpha = st.slider("Alpha (α)", min_value=0.01, max_value=0.99, value=0.3, step=0.01, key="ses_a")
    
    try:
        result = fc.ses_forecast(series, periods=int(periods), alpha=float(alpha))
        render_forecast_output(series, result, "SES", int(periods), "ses_forecast.xlsx")
    except Exception as e:
        st.error(f"Error: {e}")

def holt_tab(series):
    st.subheader("Holt's Linear Trend")
    periods = st.number_input("Forecast Periods", value=12, min_value=1, max_value=36, key="holt_p")
    
    try:
        result = fc.holt_forecast(series, periods=int(periods))
        render_forecast_output(series, result, "Holt's Linear", int(periods), "holt_forecast.xlsx")
    except Exception as e:
        st.error(f"Error: {e}")

def hw_tab(series):
    st.subheader("Holt-Winters (Triple Exponential Smoothing)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        periods = st.number_input("Forecast Periods", value=12, min_value=1, max_value=36, key="hw_p")
    with col2:
        sp = st.number_input("Seasonal Periods", value=12, min_value=4, max_value=52, key="hw_sp")
    with col3:
        trend = st.selectbox("Trend", ["add", "mul", "None"], key="hw_t")
    with col4:
        seasonal = st.selectbox("Seasonal", ["add", "mul"], key="hw_s")
        
    try:
        t = None if trend == "None" else trend
        result = fc.holtwinters_forecast(series, periods=int(periods), seasonal_periods=int(sp), trend=t, seasonal=seasonal)
        render_forecast_output(series, result, "Holt-Winters", int(periods), "hw_forecast.xlsx")
    except Exception as e:
        st.error(f"Error: {e}")

def sarima_tab(series):
    st.subheader("SARIMA")
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        periods = st.number_input("Forecast Periods", value=12, min_value=1, max_value=36, key="sar_p")
    with c2:
        sp = st.number_input("Seasonal Periods (s)", value=12, min_value=4, max_value=52, key="sar_s")
    
    st.markdown("**Order (p, d, q)**")
    c1, c2, c3 = st.columns(3)
    p = c1.number_input("p", value=1, min_value=0, max_value=5, key="sar_p_val")
    d = c2.number_input("d", value=1, min_value=0, max_value=5, key="sar_d_val")
    q = c3.number_input("q", value=1, min_value=0, max_value=5, key="sar_q_val")
    
    st.markdown("**Seasonal Order (P, D, Q)**")
    c4, c5, c6 = st.columns(3)
    P = c4.number_input("P", value=1, min_value=0, max_value=5, key="sar_P_val")
    D = c5.number_input("D", value=1, min_value=0, max_value=5, key="sar_D_val")
    Q = c6.number_input("Q", value=1, min_value=0, max_value=5, key="sar_Q_val")
    
    st.caption(f"SARIMA ({p},{d},{q})×({P},{D},{Q},{sp})")
    
    if st.button("🚀 Run SARIMA", key="sar_btn"):
        with st.spinner("Running SARIMA..."):
            try:
                order = (p, d, q)
                seasonal_order = (P, D, Q, sp)
                result = fc.sarima_forecast(series, periods=int(periods), order=order, seasonal_order=seasonal_order)
                
                render_forecast_output(series, result, f"SARIMA {order}×{seasonal_order}", int(periods), "sarima_forecast.xlsx")
                if "summary" in result:
                    with st.expander("📄 Model Summary"):
                        st.text(result["summary"])
            except Exception as e:
                st.error(f"SARIMA Error: {e}")

def auto_arima_tab(series):
    st.subheader("Auto ARIMA — Automatic Order Selection")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        periods = st.number_input("Forecast Periods", value=12, min_value=1, max_value=36, key="aa_p")
    with c2:
        m = st.number_input("Seasonal Period (m)", value=12, min_value=1, max_value=52, key="aa_m")
    with c3:
        criterion = st.selectbox("Information Criterion", ["aic", "bic", "aicc", "oob"], key="aa_c")
    with c4:
        include_seasonal = st.checkbox("Include Seasonal", value=True, key="aa_s")
        stepwise = st.checkbox("Stepwise Search", value=True, key="aa_sw")
        
    if st.button("🚀 Run Auto ARIMA", key="aa_btn"):
        with st.spinner("Running Auto ARIMA..."):
            try:
                result = fc.auto_arima_forecast(
                    series, periods=int(periods), seasonal=include_seasonal, 
                    m=int(m), stepwise=stepwise, information_criterion=criterion
                )
                o, so = result["order"], result["seasonal_order"]
                order_str = f"ARIMA({o[0]},{o[1]},{o[2]})×({so[0]},{so[1]},{so[2]},{so[3]})" if include_seasonal else f"ARIMA({o[0]},{o[1]},{o[2]})"
                st.success(f"✅ Best order: {order_str}")
                
                render_forecast_output(series, result, f"Auto ARIMA {order_str}", int(periods), "auto_arima_forecast.xlsx")
                if "summary" in result:
                    with st.expander("📄 Model Summary"):
                        st.text(result["summary"])
            except Exception as e:
                st.error(f"Auto ARIMA Error: {e}")

def decomp_tab(series):
    st.subheader("Seasonal Decomposition")
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        sp = st.number_input("Seasonal Periods", value=12, min_value=4, max_value=52, key="dec_sp")
    with c2:
        model = st.selectbox("Model", ["additive", "multiplicative"], key="dec_m")
        
    try:
        clean = series.dropna()
        if len(clean) < int(sp) * 2:
            st.warning(f"⚠️ Need at least {int(sp)*2} observations.")
        else:
            decomp = fc.decompose_series(clean, model=model, period=int(sp))
            fig = make_subplots(rows=4, cols=1, subplot_titles=["Observed", "Trend", "Seasonal", "Residuals"],
                                shared_xaxes=True, vertical_spacing=0.07)
                                
            colors = [COLORS["actual"], COLORS["trend"], COLORS["seasonal"], COLORS["resid"]]
            components = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]
            
            for i, (comp, color) in enumerate(zip(components, colors)):
                fig.add_trace(go.Scatter(x=comp.index, y=comp.values, mode="lines",
                                         line=dict(color=color, width=1.8), showlegend=False),
                              row=i+1, col=1)
                              
            fig.update_layout(height=680, margin=dict(t=30, b=30, l=50, r=20), hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#a5b4fc"),
            )
            st.plotly_chart(fig, use_container_width=True)
            
            strength_t = max(0.0, 1 - decomp.resid.var() / (decomp.trend.dropna() + decomp.resid.dropna()).var())
            strength_s = max(0.0, 1 - decomp.resid.var() / (decomp.seasonal + decomp.resid.dropna()).var())
            
            display_metrics({
                "TREND STRENGTH": f"{strength_t:.3f}",
                "SEASONAL STRENGTH": f"{strength_s:.3f}",
                "RESIDUAL STD": f"{decomp.resid.std():.3f}"
            })
    except Exception as e:
        st.error(f"Decomposition error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Forecasting App", layout="wide", page_icon="📈")
    
    st.title("📈 Time Series Forecasting")
    st.markdown("Select a report tab below to configure and view forecasts.")

    # Sidebar
    st.sidebar.title("📈 Forecasting App")
    st.sidebar.divider()

    # Template
    with st.sidebar.expander("📄 TEMPLATE", expanded=False):
        st.download_button(
            "⬇️ Download CSV Template", 
            data=blank_template_bytes(), 
            file_name="forecast_template.csv", 
            mime="text/csv",
            use_container_width=True
        )
        st.caption("`date` · `brand` · `sub_brand` · `qty`")
        
    st.sidebar.divider()

    # Data Source
    st.sidebar.caption("📂 DATA SOURCE")
    data_source = st.sidebar.radio("Source", ["Use Sample Data", "Upload File"], label_visibility="collapsed")
    df = None
    if data_source == "Use Sample Data":
        df = load_sample()
        st.sidebar.info(f"ℹ Sample: {len(df)} rows | 2019–2023")
    else:
        uploaded_file = st.sidebar.file_uploader("📁 Drag & drop or click to upload CSV / Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df = parse_uploaded_st(uploaded_file)
                st.sidebar.success(f"✔ {len(df)} rows loaded from {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

    st.sidebar.divider()

    if df is not None:
        # Filters
        st.sidebar.caption("🏷️ FILTER")
        brand_opts = sorted(df["brand"].dropna().unique().tolist())
        brand = st.sidebar.selectbox("Brand", brand_opts)
        
        subbrand_opts = sorted(df.loc[df["brand"] == brand, "sub_brand"].dropna().unique().tolist()) if brand else []
        sub_brand = st.sidebar.selectbox("Sub-Brand", subbrand_opts)
        
        freq_label = st.sidebar.selectbox("Frequency", list(FREQ_MAP.keys()))
        freq = FREQ_MAP[freq_label]
        
        if brand and sub_brand and freq:
            # Filter badges
            mask = (df["brand"] == brand) & (df["sub_brand"] == sub_brand)
            n_obs = mask.sum()
            
            st.markdown(
                f"**🏷️ Brand:** `{brand}`  |  **📦 Sub-Brand:** `{sub_brand}`  |  **📅 Observations:** `{n_obs}`"
            )

            try:
                series = get_series(df, brand, sub_brand, freq)
            except Exception as e:
                st.error(f"❌ Error filtering series: {e}")
                st.stop()
                
            # Tabs
            tabs = st.tabs([
                "📊 Overview", "📉 MA", "🔵 SES", "📐 Holt", 
                "❄️ Holt-Winters", "🤖 SARIMA", "🔮 Auto ARIMA", "🔀 Decomp"
            ])
            
            with tabs[0]:
                overview_tab(df, series, brand, sub_brand)
            with tabs[1]:
                ma_tab(series)
            with tabs[2]:
                ses_tab(series)
            with tabs[3]:
                holt_tab(series)
            with tabs[4]:
                hw_tab(series)
            with tabs[5]:
                sarima_tab(series)
            with tabs[6]:
                auto_arima_tab(series)
            with tabs[7]:
                decomp_tab(series)
        else:
            st.warning("⚠️ Please select Brand / Sub-Brand to continue.")
            
    else:
        st.warning("⚠️ Please load Sample Data or upload a valid file in the sidebar.")
        
    st.divider()
    st.caption("Python Forecasting App · Template: `date | brand | sub_brand | qty`")

if __name__ == "__main__":
    main()
