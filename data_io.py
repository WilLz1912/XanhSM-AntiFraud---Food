"""
data_io.py — Shared, cached data loaders used by app.py and pages/*.py.

orders_full.parquet expands to ~350MB in memory once loaded as a DataFrame.
Before this module existed, app.py / pages/3 / pages/5 each had their own
st.cache_data-wrapped loader for the same file, so Streamlit kept up to 3
independent ~350MB copies in memory at once — enough to exceed the ~1GB
RAM ceiling on Streamlit Community Cloud's free tier if a demo session
visited all three pages. Importing one shared cached function here means
Streamlit caches a single copy, keyed on this one function object.
"""

import os

import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
ORDERS_FULL_PATH = os.path.join(DATA_DIR, "orders_full.parquet")


@st.cache_data
def load_orders_full() -> pd.DataFrame:
    return pd.read_parquet(ORDERS_FULL_PATH)
