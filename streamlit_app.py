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
It calculates **Value at Risk (VaR)** and projects thousands of potential future price paths
based on historical volatility and correlations.
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("Configuration")

# Input for Tickers and Weights
default_tickers = "SPY, TLT, GLD"
default_weights = "0.5, 0.3, 0.2"

ticker_input = st.sidebar.text_input("Enter Tickers (comma separated)", default_tickers)
weights_input = st.sidebar.text_input("Enter Weights (comma separated)", default_weights)

# Simulation Parameters
initial_investment = st.sidebar.number_input("Initial Investment ($)", value=10000, step=1000)
years = st.sidebar.slider("Years to Forecast", min_value=1, max_value=10, value=1)
simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)

# --- Helper Function to Process Inputs ---
def process_inputs(ticker_str, weight_str):
    tickers = [t.strip().upper() for t in ticker_str.split(",")]
    try:
        weights = np.array([float(w) for w in weight_str.split(",")])
    except ValueError:
        return tickers, None
    
    # Validation: Check lengths match
    if len(tickers) != len(weights):
        st.error(f"Error: You provided {len(tickers)} tickers but {len(weights)} weights.")
        return None, None
    
    # Validation: Normalize weights if they don't sum to 1
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
                # 1. Fetch Data (5 years of history for robust stats)
                data = yf.download(tickers, period="5y")['Close']
                
                # Check if data is empty or missing columns
                if data.empty or data.shape[1] != len(tickers):
                    st.error("Error: Could not download data. Check ticker symbols.")
                    st.stop()

                # 2. Calculate Statistics
                log_returns = np.log(1 + data.pct_change()).dropna()
                mean_returns = log_returns.mean().to_numpy()
                std_devs = log_returns.std().to_numpy()
                corr_matrix = log_returns.corr().to_numpy()
                
                # 3. Cholesky Decomposition
                L = np.linalg.cholesky(corr_matrix)
                
                # 4. Run Simulation Loop
                days = 252 * years
                num_assets = len(tickers)
                portfolio_sims = np.full((days, simulations), 0.0)
                
                # Get latest prices as starting point
                current_prices = data.iloc[-1].to_numpy()

                for m in range(simulations):
                    # Generate correlated random variables
                    Z_uncorrelated = np.random.normal(0, 1, (days, num_assets))
                    Z_correlated = np.dot(L, Z_uncorrelated.T).T 
                    
                    # Reconstruct price paths
                    asset_paths = np.zeros((days, num_assets))
                    asset_paths[0] = current_prices
                    
                    for t in range(1, days):
                        drift = mean_returns - 0.5 * std_devs**2
                        diffusion = std_devs * Z_correlated[t]
                        asset_paths[t] = asset_paths[t-1] * np.exp(drift + diffusion)
                    
                    # Calculate Portfolio Value for this path
                    shares = (initial_investment * weights) / asset_paths[0]
                    portfolio_sims[:, m] = np.dot(asset_paths, shares)

                # --- Visualization Section ---
                
                # Metrics Calculation
                final_values = portfolio_sims[-1, :]
                expected_val = np.mean(final_values)
                var_95 = np.percentile(final_values, 5)
                return_pct = ((expected_val - initial_investment) / initial_investment) * 100
                
                # Display Metrics
                st.subheader("Results Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Portfolio Value", f"${expected_val:,.2f}", f"{return_pct:.1f}%")
                col2.metric("Value at Risk (95%)", f"${var_95:,.2f}", delta_color="inverse", help="5% chance the value falls below this number.")
                col3.metric("Worst Case Scenario", f"${np.min(final_values):,.2f}", delta_color="inverse")

                # Layout: Two columns for charts
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.subheader(f"Projected Paths ({simulations} Scenarios)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(portfolio_sims, linewidth=1, alpha=0.1, color='blue') # Low alpha for "spaghetti" effect
                    ax.set_xlabel("Trading Days")
                    ax.set_ylabel("Portfolio Value ($)")
                    ax.set_title(f"Monte Carlo Simulation: {years} Year Horizon")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

                with chart_col2:
                    st.subheader("Probability Distribution")
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.hist(final_values, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
                    ax2.axvline(expected_val, color='red', linestyle='--', linewidth=2, label=f"Mean: ${expected_val:,.0f}")
                    ax2.axvline(var_95, color='orange', linestyle='--', linewidth=2, label=f"VaR 95%: ${var_95:,.0f}")
                    ax2.set_xlabel("Final Portfolio Value ($)")
                    ax2.set_ylabel("Frequency")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)

            except Exception as e:
                st.error(f"An error occurred during simulation: {e}")
