# app/components/analytics.py

import streamlit.components.v1 as components


def umami_tracker():
    """Inject Umami analytics tracking script"""
    tracking_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <script defer src="https://cloud.umami.is/script.js" data-website-id="8de8dbc1-d49c-49a1-a42a-05916ad3e1a7"></script>
    </head>
    <body></body>
    </html>
    """
    components.html(tracking_html, height=0, width=0)
