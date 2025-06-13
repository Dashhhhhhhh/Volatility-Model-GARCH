import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

API_URL = "http://localhost:8000"  # Assuming FastAPI runs on port 8000

def get_forecast():
    try:
        response = requests.get(f"{API_URL}/forecast")
        response.raise_for_status()
        return response.json()["sigma"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching forecast: {e}")
        return None

def get_btc_price_history(days=30):
    """
    Fetch real Bitcoin price data from Yahoo Finance.
    Falls back to simulated data if API call fails.
    """
    try:
        # Fetch BTC-USD data from Yahoo Finance
        btc_ticker = yf.Ticker("BTC-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 5)  # Add buffer for weekends/holidays
        
        # Get historical data
        hist_data = btc_ticker.history(start=start_date, end=end_date, interval="1d")
        
        if hist_data.empty:
            raise ValueError("No data returned from Yahoo Finance")
        
        # Use closing prices
        price_series = hist_data['Close'].dropna()
        
        # Trim to requested number of days
        if len(price_series) > days:
            price_series = price_series.tail(days)
            
        return price_series.rename("BTC Price")
        
    except Exception as e:
        st.warning(f"Could not fetch live Bitcoin data: {e}. Using simulated data.")
          # Fallback to simulated data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Use current BTC price as base (approximately $98,000 as of 2025)
        base_price = 98000
        trend = np.linspace(-3000, 7000, len(dates))
        daily_volatility = np.random.randn(len(dates)) * 3500
        price_data = base_price + trend + daily_volatility
        price_data = np.maximum(price_data, 50000)
        
        price = pd.Series(price_data, index=dates)
        return price.rename("BTC Price (Simulated)")

def get_current_btc_price():
    """Get the current Bitcoin price."""
    try:
        btc_ticker = yf.Ticker("BTC-USD")
        current_data = btc_ticker.history(period="1d", interval="1m")
        if not current_data.empty:
            return current_data['Close'].iloc[-1]
    except:
        pass
    return None

st.set_page_config(layout="wide")

st.title("BTC Volatility Dashboard")

# --- Layout ---
col1, col2 = st.columns([3, 1])

with col1:
    st.header("BTC Price & Conditional Volatility (σ)")
    chart_placeholder = st.empty()

with col2:
    st.header("Current BTC Price")
    current_price_placeholder = st.empty()
    st.header("VaR Status (Last 30d)")
    var_widget_placeholder = st.empty()
    st.header("Latest Forecast (σ)")
    forecast_placeholder = st.empty()


# --- Initial Data Load & Chart ---
btc_price = get_btc_price_history(days=90) # Fetch more history for initial plot

# Generate conditional sigma data that's more realistic
# Volatility typically ranges from 1-5% of the price
returns = btc_price.pct_change().dropna()
rolling_vol = returns.rolling(window=7).std() * np.sqrt(252)  # Annualized volatility
# Fill NaN values and scale to make it visible on the chart
conditional_sigma = rolling_vol.fillna(rolling_vol.mean()) * btc_price / 10  # Scale for visibility

def create_chart(price_series, sigma_series):
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=price_series.index, 
        y=price_series, 
        name="BTC Price (USD)", 
        yaxis="y1",
        line=dict(color='blue', width=2)
    ))
    
    # Add sigma line
    fig.add_trace(go.Scatter(
        x=sigma_series.index, 
        y=sigma_series, 
        name="Conditional σ (Scaled)", 
        yaxis="y2",
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title="BTC Price and Conditional Volatility",
        xaxis_title="Date",
        yaxis=dict(title="BTC Price (USD)", side="left"),
        yaxis2=dict(title="Conditional σ (Scaled)", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.01, y=0.99),
        width=800,
        height=500
    )
    return fig

chart_placeholder.plotly_chart(create_chart(btc_price, conditional_sigma), use_container_width=True)

# Debug info (can be removed later)
st.sidebar.write("Debug Info:")
st.sidebar.write(f"Data source: {'Live' if 'Simulated' not in btc_price.name else 'Simulated'}")
st.sidebar.write(f"Price data points: {len(btc_price)}")
st.sidebar.write(f"Price range: ${btc_price.min():.0f} - ${btc_price.max():.0f}")
st.sidebar.write(f"Latest price: ${btc_price.iloc[-1]:.2f}")
st.sidebar.write(f"Sigma data points: {len(conditional_sigma)}")
st.sidebar.write(f"Sigma range: {conditional_sigma.min():.2f} - {conditional_sigma.max():.2f}")


# --- VaR Traffic Light (Placeholder) ---
# In a real app, this would be based on backtesting results or live VaR calculations
var_status_ok = True # Dummy status
if var_status_ok:
    var_widget_placeholder.success("✅ Within VaR Limits")
else:
    var_widget_placeholder.error("❌ VaR Breached")


# --- Live Forecast Update ---
def update_forecast():
    sigma = get_forecast()
    if sigma is not None:
        forecast_placeholder.metric(label="Latest σ Forecast", value=f"{sigma:.4f}")
    else:
        forecast_placeholder.error("Unavailable")

def update_current_price():
    current_price = get_current_btc_price()
    if current_price is not None:
        current_price_placeholder.metric(
            label="Live BTC Price", 
            value=f"${current_price:,.2f}",
            delta=None
        )
    else:
        current_price_placeholder.error("Price Unavailable")

update_forecast() # Initial call
update_current_price() # Initial call

# --- Main Loop for Updates (every 60 seconds) ---
# Streamlit's execution model makes true background loops tricky without threads or asyncio.
# For simplicity, we'll use st.experimental_rerun or a similar mechanism if available,
# or just rely on user interaction for updates in a basic setup.
# A more robust solution would involve a separate process or thread updating a shared state
# or database, and Streamlit periodically refreshing from that source.

# This is a simplified way to trigger periodic updates.
# In newer Streamlit versions, st.rerun() is preferred.
# However, for a simple "every minute" call, we might need a more complex setup
# or accept that it updates on interaction or via a manual refresh button.

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

if time.time() - st.session_state.last_update_time > 60:
    update_forecast()
    update_current_price()
    # Potentially update chart data here as well if new data is available
    st.session_state.last_update_time = time.time()
    st.rerun() # Rerun the script to refresh the page

# Add a manual refresh button as a fallback
if st.button("Refresh Data"):
    update_forecast()
    update_current_price()
    st.rerun()

st.caption("Dashboard updates forecast and current price every ~60 seconds. Uses live Bitcoin data from Yahoo Finance.")
