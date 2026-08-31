# ProfNITT — Nifty Portfolio Optimizer

##  Key Features

- **Live Market Data**: Scrapes real-time constituents and sector classifications directly from the National Stock Exchange (NSE) and pulls 10 years of split/dividend-adjusted price history via Yahoo Finance.
- **5 Quantitative Strategies**:
  -  **Hierarchical Risk Parity (HRP)** — Graph-theoretic machine learning clustering for robust out-of-sample risk distribution.
  -  **12M Momentum + Risk Parity** — Filters the top momentum compounders in Nifty and sizes positions using risk parity.
  -  **Max-Sharpe MVO (Mean-Variance)** — Quadratic optimization with CAPM beta anchoring and strict sector caps (e.g. max 30%).
  -  **Minimum Volatility** — Minimizes portfolio variance for defensive asset protection.
  -  **Equal-Weight Benchmark** — Naive $1/N$ baseline for realistic comparison.
- **Realistic Assumptions**:
  - **No In-Sample Overfitting**: 6-year training window (2016–2021) evaluated on an unseen 4-year out-of-sample test window (2022–today).
  - **India Risk-Free Rate**: Anchored to **6.5% p.a.** (India 91-day T-bill yield).
  - **Ledoit-Wolf Covariance Shrinkage**: Eliminates noisy covariance estimation errors.
- **Discrete Share Order Sheet**: Automatically converts fractional weights into **exact integer shares to buy** based on your total capital (₹) and live stock quotes, reporting leftover cash reserves.
- **One-Click Broker Export**: Download ready-to-execute CSV baskets for **Zerodha**, **Groww**, and **Dhan**.
- **Interactive Visualizations (Plotly)**:
  - Out-of-sample cumulative growth of ₹1 vs Nifty 50 index
  -  Underwater peak-to-trough drawdown curves
  -  Interactive stock allocation donut chart
  -  Sector concentration bar chart with constraint limits
  -  Asset correlation heatmap

---

##  Repository Structure

```
├── app.py               # Streamlit web application dashboard
├── main.py              # Standalone CLI / Python script for terminal execution
├── logo.png             # ProfNITT transparent logo with glow
├── requirements.txt     # Python package dependencies
├── .gitignore           # Git ignore rules for clean repository
└── README.md            # Project documentation & deployment guide
```

---

##  Quickstart (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/nifty-portfolio-optimizer.git
cd nifty-portfolio-optimizer
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the web application
```bash
python -m streamlit run app.py
```
*Your browser will automatically open at `http://localhost:8501`.*

---

## How to Deploy Online for FREE (2 Steps)

### Streamlit Community Cloud (Recommended)
1. Push this repository to **GitHub**.
2. Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New app"**, select your repository, set branch to `main`, and main file to `app.py`.
4. Click **"Deploy"**! You will get a live public URL (e.g. `https://profnitt-portfolio.streamlit.app`).

---

##  Mathematical Foundations

- **CAPM Expected Returns**:
  $$\mathbb{E}[R_i] = R_f + \beta_i \left(\mathbb{E}[R_m] - R_f\right)$$
- **Ledoit-Wolf Shrinkage**:
  $$\Sigma_{\text{shrunk}} = \delta F + (1 - \delta) S$$
- **Max-Sharpe Optimization**:
  $$\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}} \quad \text{s.t.} \quad \sum w_i = 1, \quad 0 \le w_i \le \text{Stock Cap}, \quad \sum_{i \in \text{Sector}} w_i \le \text{Sector Cap}$$

---

