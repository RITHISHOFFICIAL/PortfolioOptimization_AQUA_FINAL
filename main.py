"""
1.  REALISTIC RISK-FREE RATE  -- India 91-day T-bill ~6.5% p.a.
2.  TRAIN/TEST SPLIT          -- Optimise on 2016-2021, backtest on 2022-present (10-year span)
3.  ROBUST DATA FETCHING      -- Falls back gracefully on delisted / missing tickers
4.  SECTOR CONSTRAINTS        -- Max 30% per sector, Max 10% per stock
5.  MULTIPLE STRATEGIES       -- Max-Sharpe (MVO), Risk-Parity (HRP), Equal-Weight
"""

import warnings, io, requests
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works anywhere)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import date
from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# =====================================================================
# 1. FETCH LIVE NIFTY 100 CONSTITUENTS
# =====================================================================
def get_nifty100_live():
    url = "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
        df["Yahoo_Ticker"] = df["Symbol"].str.strip() + ".NS"
        sector_map = pd.Series(df["Industry"].values, index=df["Yahoo_Ticker"]).to_dict()
        return df["Yahoo_Ticker"].tolist(), sector_map, df
    except Exception as exc:
        print(f"Live fetch failed ({exc}). Using fallback list.")
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
            "VEDL.NS","ADANIPORTS.NS","BEL.NS","HAL.NS","CGPOWER.NS",
        ]
        return fallback, {t: "Unknown" for t in fallback}, pd.DataFrame({"Yahoo_Ticker": fallback})


tickers, sector_map, full_df = get_nifty100_live()
print(f"Universe: {len(tickers)} tickers.  Sample: {tickers[:5]}")


# =====================================================================
# 2. DOWNLOAD PRICE DATA   (Past 10 Years: 2016 -> today)
# =====================================================================
START_DATE  = "2016-01-01"
TODAY       = date.today().strftime("%Y-%m-%d")
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"

print(f"Downloading price data {START_DATE} -> {TODAY} ...")
raw = yf.download(tickers, start=START_DATE, end=TODAY, auto_adjust=True, progress=True)
prices_all = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

# Drop tickers with >20% missing data in the training window
missing = prices_all.loc[:TRAIN_END].isnull().mean()
bad     = missing[missing > 0.20].index.tolist()
print(f"Dropped {len(bad)} tickers with >20% missing data in train window: {bad}")

good_tickers = [t for t in prices_all.columns if t not in bad]
prices_all   = prices_all[good_tickers].ffill()
train_prices = prices_all.loc[:TRAIN_END].dropna(how="all")
test_prices  = prices_all.loc[TEST_START:].dropna(how="all")
cleaned_sector_map = {t: sector_map.get(t, "Unknown") for t in good_tickers}

print(f"Stocks retained: {len(good_tickers)}")
print(f"Train: {train_prices.index[0].date()} -> {train_prices.index[-1].date()} ({len(train_prices)} rows)")
print(f"Test : {test_prices.index[0].date()} -> {test_prices.index[-1].date()} ({len(test_prices)} rows)")


# =====================================================================
# 3. BENCHMARK (NIFTY 50)
# =====================================================================
bmark_raw = yf.download("^NSEI", start=START_DATE, end=TODAY, auto_adjust=True, progress=False)
if isinstance(bmark_raw.columns, pd.MultiIndex):
    bmark_px = bmark_raw["Close"].squeeze()
elif "Close" in bmark_raw:
    bmark_px = bmark_raw["Close"].squeeze()
else:
    bmark_px = bmark_raw.squeeze()

bmark_ret_train = bmark_px.loc[:TRAIN_END].pct_change().dropna()
bmark_ret_test  = bmark_px.loc[TEST_START:].pct_change().dropna()


# =====================================================================
# 4. CONSTANTS & HELPERS
# =====================================================================
TRADING_DAYS = 252
RF = 0.065          # India 91-day T-bill ~ 6.5% p.a.

train_returns = train_prices.pct_change().dropna()
test_returns  = test_prices.pct_change().dropna()

def portfolio_stats(weights_dict, daily_returns, rf=RF):
    tickers_in = [t for t in weights_dict if t in daily_returns.columns]
    w  = np.array([weights_dict[t] for t in tickers_in])
    r  = daily_returns[tickers_in].values
    pd_ = r @ w
    ann_ret = pd_.mean() * TRADING_DAYS
    ann_vol = pd_.std()  * np.sqrt(TRADING_DAYS)
    return ann_ret, ann_vol, (ann_ret - rf) / ann_vol

def cum_returns(weights_dict, daily_ret_df):
    tickers_in = [t for t in weights_dict if t in daily_ret_df.columns]
    w  = np.array([weights_dict[t] for t in tickers_in])
    pd_ = daily_ret_df[tickers_in].values @ w
    return (1 + pd.Series(pd_, index=daily_ret_df.index)).cumprod()


# =====================================================================
# 5. EXPECTED RETURNS (CAPM) & COVARIANCE (LEDOIT-WOLF)
# =====================================================================
#
# WHY NOT JUST USE HISTORICAL MEAN?
# Historical mean is dominated by the train period's luck/cycle.
# CAPM ties each stock's expected return to its market-beta, which
# is more stable across regimes.
#
mu_capm = expected_returns.capm_return(
    prices        = train_prices,
    market_prices = bmark_px.loc[:TRAIN_END],
    risk_free_rate= RF,
    frequency     = TRADING_DAYS,
)
S = risk_models.CovarianceShrinkage(train_prices, frequency=TRADING_DAYS).ledoit_wolf()

print(f"CAPM mu  -- mean: {mu_capm.mean()*100:.1f}%  std: {mu_capm.std()*100:.1f}%")


# =====================================================================
# 6. STRATEGY 1 -- MAX-SHARPE MVO  (with sector + stock caps)
# =====================================================================
unique_sectors   = set(cleaned_sector_map.values())
sector_lower_map = {s: 0.00 for s in unique_sectors}
sector_upper_map = {s: 0.30 for s in unique_sectors}   # 30% max per sector

ef = EfficientFrontier(mu_capm, S, weight_bounds=(0, 0.10))   # 10% max per stock
ef.add_sector_constraints(cleaned_sector_map, sector_lower=sector_lower_map, sector_upper=sector_upper_map)
_ = ef.max_sharpe(risk_free_rate=RF)
weights_mvo = ef.clean_weights()

ri, vi, sri   = portfolio_stats(weights_mvo, train_returns)
ro, vo, sro   = portfolio_stats(weights_mvo, test_returns)

print("\n" + "="*60)
print("STRATEGY 1: MAX-SHARPE MVO  (CAPM + Sector Constraints)")
print("="*60)
print(f"  IN-SAMPLE  (train)  Return={ri*100:.1f}%  Vol={vi*100:.1f}%  Sharpe={sri:.2f}")
print(f"  OUT-OF-SAMPLE (test)  Return={ro*100:.1f}%  Vol={vo*100:.1f}%  Sharpe={sro:.2f}")

top_mvo = pd.Series(weights_mvo).sort_values(ascending=False)
top_mvo_nz = top_mvo[top_mvo > 0.0001]
print(f"\n  Top {len(top_mvo_nz)} holdings:")
for tkr, w in top_mvo_nz.head(15).items():
    sec = cleaned_sector_map.get(tkr, "?")
    print(f"    {tkr:<22s}  {w*100:5.1f}%   [{sec}]")


# =====================================================================
# 7. STRATEGY 2 -- HIERARCHICAL RISK PARITY
# =====================================================================
hrp = HRPOpt(train_returns)
hrp.optimize()
weights_hrp = hrp.clean_weights()

ri2, vi2, sri2 = portfolio_stats(weights_hrp, train_returns)
ro2, vo2, sro2 = portfolio_stats(weights_hrp, test_returns)

print("\n" + "="*60)
print("STRATEGY 2: HIERARCHICAL RISK PARITY (HRP)")
print("="*60)
print(f"  IN-SAMPLE  (train)  Return={ri2*100:.1f}%  Vol={vi2*100:.1f}%  Sharpe={sri2:.2f}")
print(f"  OUT-OF-SAMPLE (test)  Return={ro2*100:.1f}%  Vol={vo2*100:.1f}%  Sharpe={sro2:.2f}")


# =====================================================================
# 8. STRATEGY 3 -- EQUAL-WEIGHT
# =====================================================================
n = len(good_tickers)
weights_ew = {t: 1.0/n for t in good_tickers}

ri3, vi3, sri3 = portfolio_stats(weights_ew, train_returns)
ro3, vo3, sro3 = portfolio_stats(weights_ew, test_returns)

print("\n" + "="*60)
print("STRATEGY 3: EQUAL-WEIGHT")
print("="*60)
print(f"  IN-SAMPLE  (train)  Return={ri3*100:.1f}%  Vol={vi3*100:.1f}%  Sharpe={sri3:.2f}")
print(f"  OUT-OF-SAMPLE (test)  Return={ro3*100:.1f}%  Vol={vo3*100:.1f}%  Sharpe={sro3:.2f}")


# =====================================================================
# 9. SUMMARY TABLE
# =====================================================================
def fmt_pct(x):   return f"{float(np.squeeze(x))*100:.1f}%"
def fmt_sr(x):    return f"{float(np.squeeze(x)):.2f}"

nifty_ri  = float(np.squeeze(bmark_ret_train.mean())) * TRADING_DAYS
nifty_vi  = float(np.squeeze(bmark_ret_train.std())) * np.sqrt(TRADING_DAYS)
nifty_sri = (nifty_ri - RF) / nifty_vi
nifty_ro  = float(np.squeeze(bmark_ret_test.mean())) * TRADING_DAYS
nifty_vo  = float(np.squeeze(bmark_ret_test.std())) * np.sqrt(TRADING_DAYS)
nifty_sro = (nifty_ro - RF) / nifty_vo

rows = [
    ("Max-Sharpe MVO", ri,  vi,  sri,  ro,  vo,  sro),
    ("Risk-Parity HRP",ri2, vi2, sri2, ro2, vo2, sro2),
    ("Equal-Weight",   ri3, vi3, sri3, ro3, vo3, sro3),
    ("Nifty 50",      nifty_ri, nifty_vi, nifty_sri,
                      nifty_ro, nifty_vo, nifty_sro),
]

print("\n" + "="*85)
print(f"{'Strategy':<20} {'IS Ret':>7} {'IS Vol':>7} {'IS SR':>6}  {'OOS Ret':>7} {'OOS Vol':>7} {'OOS SR':>6}")
print("-"*85)
for r in rows:
    print(f"{r[0]:<20} {fmt_pct(r[1]):>7} {fmt_pct(r[2]):>7} {fmt_sr(r[3]):>6}  "
          f"{fmt_pct(r[4]):>7} {fmt_pct(r[5]):>7} {fmt_sr(r[6]):>6}")
print("="*85)
print(f"  Risk-free rate: {RF*100:.1f}% p.a.  |  IS = In-sample (train 2016-2021)  |  OOS = Out-of-sample (test 2022-today)")


# =====================================================================
# 10. CHARTS
# =====================================================================
OUTDIR = r"c:\Users\vijay.000\Documents\Portfolio Optimization"

# -- Cumulative Returns (OOS) -----------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
bmark_cum = (1 + bmark_ret_test.loc[test_returns.index[0]:]).cumprod().squeeze()

for label, w in [("Max-Sharpe MVO", weights_mvo),
                 ("Risk-Parity HRP", weights_hrp),
                 ("Equal-Weight",    weights_ew)]:
    cr = cum_returns(w, test_returns)
    ax.plot(cr.index, cr.values, lw=2, label=label)
ax.plot(bmark_cum.index, bmark_cum.values, lw=2, ls="--", color="black", label="Nifty 50")
ax.set_title("Out-of-Sample Cumulative Return  (2022 to today)", fontsize=14, fontweight="bold")
ax.set_ylabel("Portfolio Value  (Rs 1 invested)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\oos_cumulative_returns.png", dpi=150)
plt.close()
print("Saved: oos_cumulative_returns.png")

# -- Drawdown Chart (OOS) --------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
for label, w in [("Max-Sharpe MVO", weights_mvo),
                 ("Risk-Parity HRP", weights_hrp),
                 ("Equal-Weight",    weights_ew)]:
    cr = cum_returns(w, test_returns)
    dd = (cr / cr.cummax() - 1) * 100
    ax.plot(dd.index, dd.values, lw=1.5, label=label)
bmark_cr  = (1 + bmark_ret_test.loc[test_returns.index[0]:]).cumprod().squeeze()
bmark_dd_ = (bmark_cr / bmark_cr.cummax() - 1) * 100
ax.plot(bmark_dd_.index, bmark_dd_.values, ls="--", lw=1.5, color="black", label="Nifty 50")
ax.set_title("Drawdown from Peak  (OOS: 2022 to today)", fontsize=14, fontweight="bold")
ax.set_ylabel("Drawdown (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\oos_drawdown.png", dpi=150)
plt.close()
print("Saved: oos_drawdown.png")

# -- MVO Allocation Pie -------------------------------------------------
top_h = {k: v for k, v in weights_mvo.items() if v > 0.005}
other = 1 - sum(top_h.values())
if other > 0.001:
    top_h["Other"] = other

fig, ax = plt.subplots(figsize=(10, 7))
ax.pie(list(top_h.values()), labels=list(top_h.keys()), autopct="%1.1f%%",
       startangle=140, textprops={"fontsize": 8})
ax.set_title("Max-Sharpe Portfolio -- Allocation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\mvo_allocation_pie.png", dpi=150)
plt.close()
print("Saved: mvo_allocation_pie.png")

# -- Sector Allocation Bar ----------------------------------------------
def sec_wts(wdict, smap):
    s = pd.Series(wdict)
    s.index = [smap.get(t, "Unknown") for t in s.index]
    return s.groupby(level=0).sum().sort_values(ascending=False)

sw_mvo_s = sec_wts(weights_mvo, cleaned_sector_map)
sw_hrp_s = sec_wts(weights_hrp, cleaned_sector_map)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, sw, title in zip(axes, [sw_mvo_s, sw_hrp_s], ["Max-Sharpe MVO", "Risk-Parity HRP"]):
    bars = ax.barh(sw.index[::-1], sw.values[::-1]*100, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Allocation (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(30, color="red", ls="--", lw=1, label="30% sector cap")
    ax.legend()
    for bar, val in zip(bars, sw.values[::-1]):
        ax.text(val*100+0.3, bar.get_y()+bar.get_height()/2, f"{val*100:.1f}%", va="center", fontsize=8)
plt.suptitle("Sector Allocation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\sector_allocation.png", dpi=150)
plt.close()
print("Saved: sector_allocation.png")

# -- Correlation Heatmap ------------------------------------------------
top20 = top_mvo_nz.head(20).index.tolist()
corr  = train_returns[top20].corr()
mask  = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            linewidths=0.4, annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation Matrix -- Top-20 MVO Holdings (train)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTDIR}\\correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

