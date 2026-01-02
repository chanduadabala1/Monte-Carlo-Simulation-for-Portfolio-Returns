import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- App Config ---
st.set_page_config(page_title="Indian Portfolio Monte Carlo", layout="wide")
st.title("🇮🇳 Indian Market Portfolio Simulator")

# --- Setup Tickers & Weights ---
# RELIANCE (Anchor), ZOMATO (Growth), TCS (IT Stability), GOLDBEES (Hedge)
default_tickers = ['RELIANCE.NS', 'ZOMATO.NS', 'TCS.NS', 'GOLDBEES.NS']
default_weights = [0.40, 0.20, 0.20, 0.20]

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Settings")
investment = st.sidebar.number_input("Initial Investment (₹)", value=100000)
simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 1000)
years = st.sidebar.slider("Years to Forecast", 1, 5, 1)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Analyzing Market DNA..."):
        # 1. Fetch Historical Data
        data = yf.download(default_tickers, period="3y")['Close']
        log_returns = np.log(1 + data.pct_change()).dropna()
        
        # 2. Stats & Cholesky
        mean_returns = log_returns.mean().to_numpy()
        std_devs = log_returns.std().to_numpy()
        L = np.linalg.cholesky(log_returns.corr().to_numpy())
        
        # 3. Simulation Engine
        days = 252 * years
        portfolio_sims = np.zeros((days, simulations))
        current_prices = data.iloc[-1].to_numpy()
        
        for m in range(simulations):
            # Correlated Random Noise
            Z = np.dot(L, np.random.normal(0, 1, (len(default_tickers), days))).T
            
            # GBM Price Paths
            drift = mean_returns - 0.5 * std_devs**2
            paths = np.zeros((days, len(default_tickers)))
            paths[0] = current_prices
            
            for t in range(1, days):
                paths[t] = paths[t-1] * np.exp(drift + std_devs * Z[t])
            
            # Calculate Portfolio Value
            shares = (investment * np.array(default_weights)) / paths[0]
            portfolio_sims[:, m] = np.dot(paths, shares)

        # --- Display Metrics ---
        final_vals = portfolio_sims[-1, :]
        expected = np.mean(final_vals)
        var_95 = np.percentile(final_vals, 5)
        
        col1, col2 = st.columns(2)
        col1.metric("Expected Final Value", f"₹{expected:,.2f}")
        col2.metric("Value at Risk (95% Confidence)", f"₹{var_95:,.2f}")

        # --- Charts ---
        st.subheader("Simulated Price Paths")
        st.line_chart(portfolio_sims[:, :50]) # Show 50 paths for clarity

        st.subheader("Distribution of Outcomes")
        fig, ax = plt.subplots()
        ax.hist(final_vals, bins=50, color='orange', edgecolor='black')
        ax.set_xlabel("Final Portfolio Value (₹)")
        st.pyplot(fig)
        
