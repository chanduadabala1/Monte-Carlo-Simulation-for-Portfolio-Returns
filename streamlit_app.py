import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Monte Carlo Portfolio Simulator", layout="wide")

st.title("📊 Monte Carlo Portfolio Simulator")
st.markdown("""
This tool simulates future portfolio performance using **Geometric Brownian Motion (GBM)**.
It calculates **Value at Risk (VaR)** and projects potential price paths based on historical volatility and correlations.
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("Configuration")

default_tickers = "SPY, TLT, GLD"
default_weights = "0.5, 0.3, 0.2"

ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", default_tickers)
weights_input = st.sidebar.text_input("Enter Weights (comma separated)", default_weights)

initial_investment = st.sidebar.number_input("Initial Investment ($)", value=10000, step=1000)
years = st.sidebar.slider("Years to Forecast", min_value=1, max_value=10, value=1)
simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)

# --- Helper Function to Process Inputs ---
def process_inputs(ticker_str, weight_str):
    tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    try:
        weights = np.array([float(w) for w in weight_str.split(",")])
    except ValueError:
        st.error("Check your weights format. Use numbers separated by commas.")
        return None, None
    
    if len(tickers) != len(weights):
        st.error(f"Error: You provided {len(tickers)} tickers but {len(weights)} weights.")
        return None, None
    
    if not np.isclose(sum(weights), 1.0):
        st.warning(f"Note: Weights sum to {sum(weights):.2f}. Normalizing to 1.0.")
        weights = weights / np.sum(weights)
        
    return tickers, weights

# --- Main Execution ---
if st.sidebar.button("Run Simulation"):
    tickers, weights = process_inputs(ticker_input, weights_input)
    
    if tickers and weights is not None:
        with st.spinner('Fetching market data and running simulations...'):
            try:
                # 1. Fetch Data
                # Fetching 5 years to ensure we have enough data for correlations
                raw_data = yf.download(tickers, period="5y")['Close']
                
                # --- DATA CLEANING (Crucial to prevent [nan, nan] error) ---
                # 1. Drop assets that are entirely NaN
                data = raw_data.dropna(axis=1, how='all')
                # 2. Fill small gaps and drop rows where any asset is missing (ensures alignment)
                data = data.ffill().dropna()
                
                if data.empty or data.shape[1] < len(tickers):
                    st.error("Error: Some tickers have no overlapping history or were not found.")
                    st.stop()

                # 2. Calculate Statistics
                # Using log returns for GBM: ln(Price_t / Price_t-1)
                log_returns = np.log(data / data.shift(1)).dropna()
                
                mean_returns = log_returns.mean().to_numpy()
                std_devs = log_returns.std().to_numpy()
                corr_matrix = log_returns.corr().to_numpy()
                
                # 3. Cholesky Decomposition
                # Adding a tiny epsilon to the diagonal ensures the matrix is positive-definite
                L = np.linalg.cholesky(corr_matrix + 1e-8 * np.eye(len(tickers)))
                
                # 4. Run Simulation Loop
                days = 252 * years
                num_assets = len(tickers)
                portfolio_sims = np.zeros((days, simulations))
                
                current_prices = data.iloc[-1].to_numpy()

                # Optimized Simulation using Vectorization
                for m in range(simulations):
                    # Generate correlated random variables
                    Z_uncorrelated = np.random.normal(0, 1, (days, num_assets))
                    Z_correlated = np.dot(L, Z_uncorrelated.T).T 
                    
                    # Geometric Brownian Motion formula
                    drift = mean_returns - 0.5 * std_devs**2
                    diffusion = std_devs * Z_correlated
                    
                    # Calculate cumulative returns for each asset
                    daily_returns = np.exp(drift + diffusion)
                    path_multipliers = np.vstack([np.ones(num_assets), daily_returns]).cumprod(axis=0)
                    
                    # Calculate asset paths ($)
                    asset_paths = current_prices * path_multipliers[1:]
                    
                    # Calculate Portfolio Value for this path
                    shares = (initial_investment * weights) / current_prices
                    portfolio_sims[:, m] = np.dot(asset_paths, shares)

                # --- Visualization Section ---
                final_values = portfolio_sims[-1, :]
                expected_val = np.mean(final_values)
                var_95 = np.percentile(final_values, 5)
                return_pct = ((expected_val - initial_investment) / initial_investment) * 100
                
                st.subheader("Results Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Portfolio Value", f"${expected_val:,.2f}", f"{return_pct:.1f}%")
                col2.metric("Value at Risk (95%)", f"${var_95:,.2f}", delta_color="inverse", help="5% chance the value falls below this number.")
                col3.metric("Worst Case Scenario", f"${np.min(final_values):,.2f}", delta_color="inverse")

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.subheader(f"Projected Paths")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(portfolio_sims, linewidth=1, alpha=0.05, color='#1f77b4') 
                    ax.set_xlabel("Trading Days")
                    ax.set_ylabel("Portfolio Value ($)")
                    ax.grid(True, alpha=0.2)
                    st.pyplot(fig)

                with chart_col2:
                    st.subheader("Probability Distribution")
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.hist(final_values, bins=50, color='skyblue', edgecolor='white', alpha=0.8)
                    ax2.axvline(expected_val, color='red', linestyle='--', label=f"Mean")
                    ax2.axvline(var_95, color='orange', linestyle='--', label=f"VaR 95%")
                    ax2.set_xlabel("Final Portfolio Value ($)")
                    ax2.set_ylabel("Frequency")
                    ax2.legend()
                    st.pyplot(fig2)

            except Exception as e:
                st.error(f"Simulation Error: {e}")
                st.info("Try checking if your tickers have overlapping historical data.")
                
