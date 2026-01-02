import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- App Config ---
st.set_page_config(page_title="Indian Portfolio Monte Carlo", layout="wide")
st.title("🇮🇳 Indian Market Portfolio Simulator")

# --- Setup Tickers & Weights ---
default_tickers = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'GOLDBEES.NS']
default_weights = [0.40, 0.20, 0.20, 0.20]

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Settings")
investment = st.sidebar.number_input("Initial Investment (₹)", value=100000)
simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 1000)
years = st.sidebar.slider("Years to Forecast", 1, 5, 1)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Analyzing Market DNA..."):
        # 1. Fetch Historical Data with Error Handling
        data = yf.download(default_tickers, period="3y")['Close']
        
        # FIX 1: Fill missing values and drop remaining NaNs
        data = data.ffill().dropna()
        
        if data.empty or data.shape[1] < len(default_tickers):
            st.error("Could not fetch complete data for all tickers. Please try again later.")
            st.stop()

        # 2. Stats & Cholesky
        log_returns = np.log(1 + data.pct_change()).dropna()
        mean_returns = log_returns.mean().to_numpy()
        std_devs = log_returns.std().to_numpy()
        
        # FIX 2: Check for correlation matrix validity
        corr_matrix = log_returns.corr().to_numpy()
        if np.isnan(corr_matrix).any():
            st.error("Statistical error: Constant returns detected. Simulation cannot proceed.")
            st.stop()
            
        L = np.linalg.cholesky(corr_matrix)
        
        # 3. Simulation Engine
        days = 252 * years
        portfolio_sims = np.zeros((days, simulations))
        current_prices = data.iloc[-1].to_numpy()
        
        for m in range(simulations):
            Z = np.dot(L, np.random.normal(0, 1, (len(default_tickers), days))).T
            drift = mean_returns - 0.5 * std_devs**2
            paths = np.zeros((days, len(default_tickers)))
            paths[0] = current_prices
            
            for t in range(1, days):
                paths[t] = paths[t-1] * np.exp(drift + std_devs * Z[t])
            
            shares = (investment * np.array(default_weights)) / paths[0]
            portfolio_sims[:, m] = np.dot(paths, shares)

        # --- 4. Final Outcomes Cleanup ---
        final_vals = portfolio_sims[-1, :]
        
        # FIX 3: Remove NaNs or Infs before plotting
        final_vals = final_vals[np.isfinite(final_vals)]
        
        if len(final_vals) == 0:
            st.error("The simulation resulted in invalid values. Check your input parameters.")
            st.stop()

        # --- Display Metrics ---
        expected = np.mean(final_vals)
        var_95 = np.percentile(final_vals, 5)
        
        col1, col2 = st.columns(2)
        col1.metric("Expected Final Value", f"₹{expected:,.2f}")
        col2.metric("Value at Risk (95%)", f"₹{var_95:,.2f}")

        # --- Charts ---
        st.subheader("Results Visualization")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Path Plot
        ax1.plot(portfolio_sims[:, :100], alpha=0.3)
        ax1.set_title("Simulated Price Paths (First 100)")
        ax1.set_ylabel("Value (₹)")

        # Histogram Plot (The part that was crashing)
        ax2.hist(final_vals, bins=50, color='orange', edgecolor='black')
        ax2.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: ₹{var_95:,.0f}')
        ax2.set_title("Distribution of Final Values")
        ax2.legend()
        
        st.pyplot(fig)
        
