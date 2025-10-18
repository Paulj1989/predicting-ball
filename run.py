# run.py

import streamlit as st

st.set_page_config(
    page_title="Predicting Ball",
    page_icon="⚽️",
    layout="wide",
)

from app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
