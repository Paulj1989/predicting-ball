# app/components/footer.py

import streamlit as st


def render():
    """Render the footer with link to homepage"""
    st.markdown(
        """
        <div class="app-footer">
            <p>Built by <a href="https://paulrjohnson.net" target="_blank">Paul Johnson</a></p>
        </div>
    """,
        unsafe_allow_html=True,
    )
