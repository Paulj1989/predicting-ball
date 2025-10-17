# app/components/analytics.py

import streamlit as st


def umami_tracker():
    """Inject Umami analytics tracking script"""
    st.html(
        """
        <script defer src="https://cloud.umami.is/script.js" data-website-id="8de8dbc1-d49c-49a1-a42a-05916ad3e1a7"></script>
        """
    )
