import io
import requests
import warnings
warnings.filterwarnings("ignore")
from datetime import date

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models, DiscreteAllocation


st.set_page_config(
    page_title="Nifty Quant Portfolio Optimizer",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0c1a30 0%, #050b14 100%);
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Glowing Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: rgba(15, 30, 56, 0.75);
        backdrop-filter: blur(10px);
        padding: 20px 22px;
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        box-shadow: 0 8px 24px -4px rgba(2, 132, 199, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 12px 28px -4px rgba(2, 132, 199, 0.3);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.02em;
        margin-top: 4px;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.08em;
    }
    
    /* Sidebar Blue Theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d213f 0%, #081528 50%, #050d1a 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.25) !important;
        box-shadow: 4px 0 24px rgba(2, 132, 199, 0.15) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #38bdf8 !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stWidgetLabel p {
        color: #93c5fd !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stNumberInput > div > div input {
        background-color: #112548 !important;
        color: #f8fafc !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within,
    section[data-testid="stSidebar"] .stNumberInput > div > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.35) !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(90deg, #0284c7 0%, #38bdf8 100%) !important;
        color: #041329 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3) !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
        color: #020d1c !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(90deg, rgba(2, 132, 199, 0.2) 0%, rgba(56, 189, 248, 0.05) 100%);
        border-left: 4px solid #38bdf8;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* Custom Badge */
    .badge-accent {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(56, 189, 248, 0.15);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 8px;
    }
    /* Logo Image Glow */
    section[data-testid="stSidebar"] img {
        filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.45));
        margin-bottom: 12px;
        transition: filter 0.3s ease;
    }
    section[data-testid="stSidebar"] img:hover {
        filter: drop-shadow(0 0 18px rgba(56, 189, 248, 0.8));
    }
</style>
""", unsafe_allow_html=True)





@st.cache_data(ttl=3600*12)
def get_nifty_constituents(universe="Nifty 50"):
    url_50 = "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv"
    url_100 = "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv"
    url = url_50 if universe == "Nifty 50" else url_100
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
        df["Yahoo_Ticker"] = df["Symbol"].str.strip() + ".NS"
        name_map = pd.Series(df["Company Name"].values, index=df["Yahoo_Ticker"]).to_dict()
        sector_map = pd.Series(df["Industry"].values, index=df["Yahoo_Ticker"]).to_dict()
        return df["Yahoo_Ticker"].tolist(), name_map, sector_map
    except Exception:
        fallback = [
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
            "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","BAJFINANCE.NS","KOTAKBANK.NS",
            "ADANIENT.NS","LT.NS","ASIANPAINT.NS","AXISBANK.NS","MARUTI.NS",
            "WIPRO.NS","HCLTECH.NS","SUNPHARMA.NS","ULTRACEMCO.NS","TITAN.NS",
            "NESTLEIND.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","COALINDIA.NS",
            "TECHM.NS","ITC.NS","BAJAJ-AUTO.NS","DRREDDY.NS","DIVISLAB.NS",
            "CIPLA.NS","GRASIM.NS","JSWSTEEL.NS","TATAMOTORS.NS","TATASTEEL.NS",
            "HINDALCO.NS","M&M.NS","BRITANNIA.NS","EICHERMOT.NS","BPCL.NS",
            "APOLLOHOSP.NS","TRENT.NS","BAJAJFINSV.NS","SBILIFE.NS","HDFCLIFE.NS",
            "VEDL.NS","ADANIPORTS.NS","BEL.NS","HAL.NS","CGPOWER.NS"
        ]
        return fallback, {t: t.replace(".NS", "") for t in fallback}, {t: "Diversified" for t in fallback}

@st.cache_data(ttl=3600*12)
def load_price_data(tickers, start_date="2016-01-01"):
    today = date.today().strftime("%Y-%m-%d")
    raw = yf.download(tickers + ["^NSEI"], start=start_date, end=today, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"]
    else:
        close_df = raw
    
    bmark_px = close_df["^NSEI"].squeeze().ffill()
    stock_px = close_df.drop(columns=["^NSEI"], errors="ignore").ffill()
    
    
    valid_cols = stock_px.isnull().mean()[lambda x: x < 0.20].index.tolist()
    stock_px = stock_px[valid_cols].dropna(how="all")
    
    return stock_px, bmark_px





import os
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)

st.sidebar.title("Portfolio Controls")

if st.sidebar.button("↻ Refresh Live Market Data"):
    st.cache_data.clear()
    st.rerun()

universe = st.sidebar.selectbox("Market Universe", ["Nifty 50", "Nifty 100"], index=0)
capital = st.sidebar.number_input("Total Investment Capital (₹)", min_value=10000, max_value=100000000, value=500000, step=25000, format="%d")
top_n = st.sidebar.slider("Number of Stocks to Pick", min_value=5, max_value=35, value=15, step=1)

strategy = st.sidebar.selectbox(
    "Optimization Strategy",
    [
        "Hierarchical Risk Parity (HRP) [Recommended]",
        "12M Momentum + Risk Parity",
        "Max-Sharpe MVO (Sector Constrained)",
        "Minimum Volatility MVO",
        "Equal-Weight Benchmark"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Constraints")
sector_cap = st.sidebar.slider("Max Sector Cap (%)", min_value=15, max_value=60, value=30, step=5) / 100.0
stock_cap = st.sidebar.slider("Max Single Stock Cap (%)", min_value=5, max_value=30, value=12, step=1) / 100.0





st.title("Nifty Portfolio Optimizer")


tickers_all, name_map, sector_map = get_nifty_constituents(universe)

with st.spinner("Fetching market quotes and computing optimal risk weights..."):
    stock_px, bmark_px = load_price_data(tickers_all, start_date="2016-01-01")
    
   
    split_date = "2021-12-31"
    train_px = stock_px.loc[:split_date]
    test_px  = stock_px.loc["2022-01-01":]
    
    train_ret = train_px.pct_change().dropna()
    test_ret  = test_px.pct_change().dropna()
    
    bmark_ret = bmark_px.pct_change().dropna()
    bmark_ret_test = bmark_ret.loc[test_ret.index[0]:]
    
    RF = 0.065
    TRADING_DAYS = 252
    


    
    mu_capm = expected_returns.capm_return(train_px, market_prices=bmark_px.loc[:split_date], risk_free_rate=RF, frequency=TRADING_DAYS)
    S = risk_models.CovarianceShrinkage(train_px, frequency=TRADING_DAYS).ledoit_wolf()
    


    
    if "Momentum" in strategy:
        mom_12m = (train_px.iloc[-1] / train_px.iloc[-252] - 1.0).sort_values(ascending=False)
        selected_tickers = mom_12m.head(top_n).index.tolist()
        hrp = HRPOpt(train_ret[selected_tickers])
        hrp.optimize()
        raw_weights = hrp.clean_weights()
    elif "HRP" in strategy:
        hrp = HRPOpt(train_ret)
        hrp.optimize()
        full_w = hrp.clean_weights()
        top_s = pd.Series(full_w).sort_values(ascending=False).head(top_n)
        raw_weights = (top_s / top_s.sum()).to_dict()
    elif "Max-Sharpe" in strategy:
        unique_sec = set(sector_map.values())
        sec_upper = {s: sector_cap for s in unique_sec}
        sec_lower = {s: 0.0 for s in unique_sec}
        ef = EfficientFrontier(mu_capm, S, weight_bounds=(0, stock_cap))
        cleaned_sec_map = {t: sector_map.get(t, "Diversified") for t in stock_px.columns}
        ef.add_sector_constraints(cleaned_sec_map, sector_lower=sec_lower, sector_upper=sec_upper)
        ef.max_sharpe(risk_free_rate=RF)
        full_w = ef.clean_weights()
        top_s = pd.Series(full_w).sort_values(ascending=False).head(top_n)
        raw_weights = (top_s / top_s.sum()).to_dict()
    elif "Minimum Volatility" in strategy:
        ef = EfficientFrontier(mu_capm, S, weight_bounds=(0, stock_cap))
        ef.min_volatility()
        full_w = ef.clean_weights()
        top_s = pd.Series(full_w).sort_values(ascending=False).head(top_n)
        raw_weights = (top_s / top_s.sum()).to_dict()
    else: # Equal-Weight
        selected_tickers = stock_px.columns[:top_n]
        raw_weights = {t: 1.0 / top_n for t in selected_tickers}
    
    weights = {k: v for k, v in raw_weights.items() if v > 0.0005}



active_tickers = list(weights.keys())
w_arr = np.array([weights[t] for t in active_tickers])
w_arr = w_arr / w_arr.sum()



port_daily_test = test_ret[active_tickers].values @ w_arr
ann_ret_oos = float(np.squeeze(port_daily_test.mean())) * TRADING_DAYS
ann_vol_oos = float(np.squeeze(port_daily_test.std())) * np.sqrt(TRADING_DAYS)
sharpe_oos  = (ann_ret_oos - RF) / ann_vol_oos




port_cum_oos = (1 + pd.Series(port_daily_test, index=test_ret.index)).cumprod()
bmark_cum_oos = (1 + bmark_ret_test).cumprod()
drawdown = (port_cum_oos / port_cum_oos.cummax() - 1)
max_dd = float(np.squeeze(drawdown.min()))



c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Expected Annual Return</div><div class="metric-val">{ann_ret_oos*100:.1f}%</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Annualized Volatility</div><div class="metric-val">{ann_vol_oos*100:.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Sharpe Ratio (Rf=6.5%)</div><div class="metric-val">{sharpe_oos:.2f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-lbl">Max Drawdown</div><div class="metric-val">{max_dd*100:.1f}%</div></div>', unsafe_allow_html=True)

st.markdown("###")




latest_prices = stock_px[active_tickers].iloc[-1]
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=capital)
allocation, leftover = da.greedy_portfolio()

order_rows = []
for t in active_tickers:
    qty = allocation.get(t, 0)
    px_val = float(latest_prices[t])
    invested = qty * px_val
    order_rows.append({
        "Ticker": t,
        "Company Name": name_map.get(t, t.replace(".NS", "")),
        "Sector": sector_map.get(t, "Diversified"),
        "Live Price": f"₹{px_val:,.2f}",
        "Target Weight": f"{weights[t]*100:.1f}%",
        "Shares to Buy": qty,
        "Total Value": f"₹{invested:,.2f}",
        "_Invested_Raw": invested
    })

order_df = pd.DataFrame(order_rows).sort_values(by="_Invested_Raw", ascending=False)
export_df = order_df.drop(columns=["_Invested_Raw"])

st.subheader("Order Execution Sheet")
st.dataframe(export_df, use_container_width=True, hide_index=True)

col_summary_1, col_summary_2 = st.columns([2, 1])
with col_summary_1:
    st.info(f" **Total Capital Allocated**: ₹{capital - leftover:,.2f} | **Cash Reserve**: ₹{leftover:,.2f}")
with col_summary_2:
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Order Basket (CSV)",
        data=csv,
        file_name=f"nifty_portfolio_orders_{date.today()}.csv",
        mime="text/csv"
    )




st.markdown("---")
tab1, tab2, tab3 = st.tabs([" Backtest & Drawdown", " Asset & Sector Allocation", "Correlation Heatmap"])

plot_layout_theme = dict(
    paper_bgcolor="rgba(15, 30, 56, 0.4)",
    plot_bgcolor="rgba(15, 30, 56, 0.4)",
    font=dict(color="#e2e8f0"),
    xaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)"),
    yaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)"),
)

with tab1:
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=port_cum_oos.index, y=port_cum_oos.values, mode='lines', name=strategy.split('[')[0].strip(), line=dict(color='#38bdf8', width=2.5)))
    fig_cum.add_trace(go.Scatter(x=bmark_cum_oos.index, y=bmark_cum_oos.values, mode='lines', name='Nifty 50 Benchmark', line=dict(color='#94a3b8', width=2, dash='dash')))
    fig_cum.update_layout(title="Out-of-Sample Growth of ₹1 (2022 to Today)", height=420, xaxis_title="Date", yaxis_title="Portfolio Multiple", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), **plot_layout_theme)
    st.plotly_chart(fig_cum, use_container_width=True)
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values*100, fill='tozeroy', name='Drawdown', line=dict(color='#f87171', width=1.5), fillcolor='rgba(248, 113, 113, 0.2)'))
    fig_dd.update_layout(title="Underwater Drawdown from Peak (%)", height=300, xaxis_title="Date", yaxis_title="Drawdown %", **plot_layout_theme)
    st.plotly_chart(fig_dd, use_container_width=True)

with tab2:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_pie = px.pie(names=[name_map.get(t, t.replace(".NS", "")) for t in active_tickers], values=[weights[t] for t in active_tickers], title="Stock Allocations", hole=0.45, color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(**plot_layout_theme)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_p2:
        sec_series = pd.Series(weights)
        sec_series.index = [sector_map.get(t, "Diversified") for t in sec_series.index]
        sec_agg = sec_series.groupby(level=0).sum().sort_values(ascending=True) * 100
        
        fig_bar = px.bar(x=sec_agg.values, y=sec_agg.index, orientation='h', title=f"Sector Distribution (Max {sector_cap*100:.0f}% Cap)", labels={'x': 'Weight (%)', 'y': 'Sector'}, color_discrete_sequence=['#0284c7'])
        fig_bar.add_vline(x=sector_cap*100, line_dash="dash", line_color="#ef4444", annotation_text="Limit", annotation_position="top right")
        fig_bar.update_layout(**plot_layout_theme)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    corr_df = train_ret[active_tickers].corr()
    fig_heat = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale="Blues", title="Asset Correlation Matrix (Training Window)")
    fig_heat.update_layout(**plot_layout_theme)
    st.plotly_chart(fig_heat, use_container_width=True)

