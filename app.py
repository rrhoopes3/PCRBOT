import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import time

presets_options = ["^SPX", "^NDX", "^RUT", "OEX", "XSP", "^VIX", "SPY", "QQQ", "IWM", "DIA", "XLE", "XLF", "XLK", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
st.set_page_config(page_title="Multi PCR Dashboard", layout="wide")
st.title("🕐 Multi PCR Dashboard")
st.markdown("**Updates every 60s (15min delayed data)**. Vol/OI from nearest expiry.")

if 'histories' not in st.session_state:
    st.session_state.histories = {}
if 'tickers' not in st.session_state:
    st.session_state.tickers = ["^SPX"]

@st.cache_data(ttl=70)
def fetch_data(ticker, expiry):
    try:
        ticker_obj = yf.Ticker(ticker)
        chain = ticker_obj.option_chain(expiry)
        puts = chain.puts
        calls = chain.calls
        if puts.empty or calls.empty:
            return None
        put_vol = puts['volume'].sum(skipna=True)
        call_vol = calls['volume'].sum(skipna=True)
        put_oi = puts['openInterest'].sum(skipna=True)
        call_oi = calls['openInterest'].sum(skipna=True)
        vol_ratio = put_vol / call_vol if call_vol > 0 else np.nan
        oi_ratio = put_oi / call_oi if call_oi > 0 else np.nan
        return {
            'vol_ratio': vol_ratio, 'oi_ratio': oi_ratio,
            'put_vol': int(put_vol) if pd.notna(put_vol) else 0, 'call_vol': int(call_vol) if pd.notna(call_vol) else 0,
            'put_oi': int(put_oi) if pd.notna(put_oi) else 0, 'call_oi': int(call_oi) if pd.notna(call_oi) else 0,
            'expiry': expiry, 'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    except Exception:
        return None

with st.sidebar:
    selected_presets = st.multiselect("Presets", presets_options, default=st.session_state.tickers)
    custom_ticker = st.text_input("Custom ticker (optional):")
    selected_tickers = list(set(t.strip().upper() for t in selected_presets + ([custom_ticker] if custom_ticker.strip() else []) if t.strip()))
    if selected_tickers:
        st.session_state.tickers = selected_tickers
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    if not st.session_state.tickers:
        st.info("👋 Select tickers above to get started!")

if st.session_state.tickers:
    tab_container = st.tabs(st.session_state.tickers)
    for i, ticker in enumerate(st.session_state.tickers):
        with tab_container[i]:
            ticker_obj = yf.Ticker(ticker)
            options = ticker_obj.options
            if not options:
                st.error(f"No options available for {ticker}")
                continue
            nearest_expiries = options[:10]
            expiry_idx = st.selectbox("Select Expiry (nearest first)", nearest_expiries, index=0, key=f"expiry_{i}")
            key = f"{ticker}_{expiry_idx}"
            data = fetch_data(ticker, expiry_idx)
            if data:
                if key not in st.session_state.histories:
                    st.session_state.histories[key] = []
                st.session_state.histories[key].append(data)
                st.session_state.histories[key] = st.session_state.histories[key][-100:]
                df = pd.DataFrame(st.session_state.histories[key])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns([1, 1, 1, 1, 1.2])
                mcol1.metric("Vol Ratio", f"{data['vol_ratio']:.2f}")
                mcol2.metric("OI Ratio", f"{data['oi_ratio']:.2f}")
                mcol3.metric("Put Vol", f"{data['put_vol']:,}")
                mcol4.metric("Call Vol", f"{data['call_vol']:,}")
                # Video indicator in upper right
                if pd.notna(data['vol_ratio']):
                    if data['vol_ratio'] > 1:
                        mcol5.video('bearish.mp4', format='video/mp4', start_time=0, autoplay=True, loop=True, muted=True)
                    elif data['vol_ratio'] < 1:
                        mcol5.video('bullish.mp4', format='video/mp4', start_time=0, autoplay=True, loop=True, muted=True)
                    else:
                        mcol5.video('neutral.mp4', format='video/mp4', start_time=0, autoplay=True, loop=True, muted=True)
                st.caption(f"Expiry: {data['expiry']}")
                fig = px.line(df, x='timestamp', y=['vol_ratio', 'oi_ratio'], title=f"{ticker} Ratio Trends")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"No data for {ticker}/{expiry_idx} (market closed or invalid?)")

# Auto-refresh
time.sleep(60)
st.rerun()