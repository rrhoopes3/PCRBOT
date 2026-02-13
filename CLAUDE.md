# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

- Run the application (handles virtual environment and dependencies): `./start.sh`
- Manual run: `source venv/bin/activate && streamlit run app.py`
- Access the dashboard at http://localhost:8501 (data available during market hours)

## High-Level Architecture

FinanceBot is a single-file Streamlit application (app.py) that creates a dashboard for monitoring put/call ratios of options for stock indices and ETFs. It uses yfinance to fetch option chain data for selected tickers and nearest expiries, calculates volume and open interest ratios, and displays metrics and trend charts in tabs. The app auto-refreshes every 60 seconds using time.sleep and st.rerun.

## Key Information from README.md

- The app is an SPX Dashboard that updates every 60s with 15min delayed data.
- Volume/OI data is from the nearest expiry.
