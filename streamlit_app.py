import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Force 'Close' to be a clean DataFrame
data = yf.download(tickers, period="5y")['Close']
if isinstance(data, pd.Series): # If only one ticker is provided
    data = data.to_frame()

# --- App Config ---
st.set_page_config(page_title="Indian Portfolio Monte Carlo", layout="wide")
st.title("🇮🇳 Indian Market Portfolio Simulator")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Settings")

# 1. Dynamic Ticker Input
ticker_input = st.sidebar.text_input(
    "Enter Ticker Symbols (comma separated)", 
    value="RELIANCE.NS, INFY.NS, TCS.NS, GOLDBEES.NS"
)
tickers = [t.strip() for t in ticker_input.split(",")]

# 2. Dynamic Weights Input
weight_input = st.sidebar.text_input(
    "Enter Weights (must sum to 1.0)", 
    value="0.40, 0.20, 0.20, 0.20"
)
try:
    weights = np.array([float(w.strip()) for w in weight_input.split(",")])
except ValueError:
    st.error("Please enter valid numerical weights.")
    st.stop()

# 3. Forecast and Simulation Settings
investment = st.sidebar.number_input("Initial Investment (₹)", value=100000)
simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)
# Increased max_value to 10 years as requested
years = st.sidebar.slider("Years to Forecast", 1, 10, 1)

if st.sidebar.button("Run Simulation"):
    # Validation: Weights must match Tickers
    if len(tickers) != len(weights):
        st.error(f"Mismatch: You have {len(tickers)} tickers but {len(weights)} weights.")
        st.stop()
    
    # Validation: Weights normalization
    if not np.isclose(weights.sum(), 1.0):
        st.warning(f"Weights sum to {weights.sum()}. Normalizing to 1.0...")
        weights = weights / weights.sum()

    with st.spinner("Fetching Market Data & Running Simulations..."):
        # 1. Fetch Historical Data
        data = yf.download(tickers, period="5y")['Close']
        data = data.ffill().dropna() # Fix for holidays/missing data
        
        if data.empty or data.shape[1] < len(tickers):
            st.error("Error fetching data. Ensure tickers are valid (e.g., RELIANCE.NS).")
            st.stop()

        # 2. Calculate Log Returns & Correlation (The "Market DNA")
        log_returns = np.log(1 + data.pct_change()).dropna()
        mean_returns = log_returns.mean().to_numpy()
        std_devs = log_returns.std().to_numpy()
        
        corr_matrix = log_returns.corr().to_numpy()
        if np.isnan(corr_matrix).any():
            st.error("Statistical error in data. Try a different date range or tickers.")
            st.stop()
            
        L = np.linalg.cholesky(corr_matrix)
        
        # 3. Simulation Engine (GBM with Cholesky)
        days = 252 * years
        portfolio_sims = np.zeros((days, simulations))
        current_prices = data.iloc[-1].to_numpy()
        
        for m in range(simulations):
            # Correlated random shocks
            Z = np.dot(L, np.random.normal(0, 1, (len(tickers), days))).T
            drift = mean_returns - 0.5 * std_devs**2
            
            paths = np.zeros((days, len(tickers)))
            paths[0] = current_prices
            
            for t in range(1, days):
                paths[t] = paths[t-1] * np.exp(drift + std_devs * Z[t])
            
            # Daily Portfolio Value calculation
            shares = (investment * weights) / paths[0]
            portfolio_sims[:, m] = np.dot(paths, shares)

        # 4. Results Cleanup
        final_vals = portfolio_sims[-1, :]
        final_vals = final_vals[np.isfinite(final_vals)] # FIX for the [nan, nan] error

        # --- Visualizations ---
        expected = np.mean(final_vals)
        var_95 = np.percentile(final_vals, 5)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Value", f"₹{expected:,.2f}")
        col2.metric("VaR (95%)", f"₹{var_95:,.2f}")
        col3.metric("Return (%)", f"{((expected/investment)-1)*100:.2f}%")

        st.divider()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: The "Spaghetti" Simulation Paths
        ax1.plot(portfolio_sims[:, :100], alpha=0.3)
        ax1.set_title(f"{years}-Year Forecast Paths (First 100 Simulations)")
        ax1.set_xlabel("Trading Days")
        ax1.set_ylabel("Portfolio Value (₹)")

        # Plot 2: Histogram of final outcomes
        ax2.hist(final_vals, bins=50, color='skyblue', edgecolor='black')
        ax2.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: ₹{var_95:,.0f}')
        ax2.axvline(expected, color='green', label=f'Mean: ₹{expected:,.0f}')
        ax2.set_title("Distribution of Possible Final Values")
        ax2.set_xlabel("Final Value (₹)")
        ax2.legend()
        
        st.pyplot(fig)
