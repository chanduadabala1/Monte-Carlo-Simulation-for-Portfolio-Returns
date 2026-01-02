!pip install streamlit
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
It automatically cleans data and handles overlapping historical periods.
""")

# --- Sidebar: User Inputs ---
st.sidebar.header("Configuration")

default_tickers = "SPY, TLT, GLD, BTC-USD"
default_weights = "0.4, 0.3, 0.2, 0.1"

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
        return None, None

    if len(tickers) != len(weights):
        st.error(f"Error: {len(tickers)} tickers vs {len(weights)} weights.")
        return None, None

    return tickers, weights

# --- Main Execution ---
if st.sidebar.button("Run Simulation"):
    tickers, weights = process_inputs(ticker_input, weights_input)

    if tickers and weights is not None:
        with st.spinner('Fetching and aligning market data...'):
            try:
                # 1. Fetch Data
                raw_data = yf.download(tickers, period="5y")['Close']

                # Handle single-ticker edge case
                if isinstance(raw_data, pd.Series):
                    raw_data = raw_data.to_frame()

                # 2. Robust Data Cleaning (Fixes the [nan, nan] and "No Overlap" errors)
                # Drop tickers that are entirely NaN
                found_tickers = raw_data.columns[~raw_data.isnull().all()].tolist()
                missing = list(set(tickers) - set(found_tickers))

                if missing:
                    st.warning(f"Skipping (No Data found): {', '.join(missing)}")

                # Filter data to only found tickers and drop rows with ANY NaNs to ensure overlap
                data = raw_data[found_tickers].dropna()

                if data.empty or len(data) < 30:
                    st.error("Error: Not enough overlapping historical data. Try older tickers or a shorter timeframe.")
                    st.stop()

                # Re-align weights based on surviving tickers
                indices_to_keep = [tickers.index(t) for t in found_tickers]
                final_weights = weights[indices_to_keep]
                final_weights = final_weights / np.sum(final_weights) # Normalize to 1.0

                # 3. Calculate Statistics
                log_returns = np.log(data / data.shift(1)).dropna()
                mean_returns = log_returns.mean().to_numpy()
                std_devs = log_returns.std().to_numpy()
                corr_matrix = log_returns.corr().to_numpy()

                # 4. Cholesky Decomposition (with stability epsilon)
                L = np.linalg.cholesky(corr_matrix + 1e-8 * np.eye(len(found_tickers)))

                # 5. Run Simulation
                days = 252 * years
                num_assets = len(found_tickers)
                portfolio_sims = np.zeros((days, simulations))
                current_prices = data.iloc[-1].to_numpy()

                for m in range(simulations):
                    Z = np.random.normal(0, 1, (days, num_assets))
                    Z_corr = np.dot(L, Z.T).T

                    drift = mean_returns - 0.5 * std_devs**2
                    diffusion = std_devs * Z_corr

                    # Compute price paths
                    daily_returns = np.exp(drift + diffusion)
                    path_multipliers = np.vstack([np.ones(num_assets), daily_returns]).cumprod(axis=0)
                    asset_paths = current_prices * path_multipliers[1:]

                    # Portfolio value calculation
                    shares = (initial_investment * final_weights) / current_prices
                    portfolio_sims[:, m] = np.dot(asset_paths, shares)

                # --- Results UI ---
                st.success(f"Successfully simulated using {len(found_tickers)} assets over {len(data)} overlapping days.")

                final_values = portfolio_sims[-1, :]
                expected_val = np.mean(final_values)
                var_95 = np.percentile(final_values, 5)

                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Value", f"${expected_val:,.2f}")
                col2.metric("VaR (95%)", f"${var_95:,.2f}", delta_color="inverse")
                col3.metric("Max Drawdown (Sim)", f"${(np.min(final_values) - initial_investment):,.2f}", delta_color="inverse")

                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots()
                    ax.plot(portfolio_sims, color='royalblue', alpha=0.05)
                    ax.set_title("Projected Portfolio Paths")
                    st.pyplot(fig)
                with c2:
                    fig2, ax2 = plt.subplots()
                    ax2.hist(final_values, bins=50, color='teal', alpha=0.7)
                    ax2.axvline(var_95, color='orange', label="VaR 95%")
                    ax2.set_title("Distribution of Final Outcomes")
                    st.pyplot(fig2)

            except Exception as e:
                st.error(f"Critical Simulation Error: {e}")
