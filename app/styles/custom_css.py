# app/styles/custom_css.py

import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styling"""
    st.markdown(
        """
    <style>
    /* Import fonts from Google Fonts with all needed weights */
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=Poppins:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Fira+Code:wght@300..700&display=swap');

    /* Root variables matching your website palette */
    :root {
        --primary-color: #43494D;
        --secondary-color: #2C3033;
        --dark-gray: #212529;
        --medium-gray: #596166;
        --light-gray: #5A6266;
        --gray-light: #F7F9FA;
        --border-color: #dee2e6;

        /* Font family variables for consistency */
        --font-sans: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-serif: 'Lora', serif;
        --font-mono: 'Fira Code', 'Consolas', 'Monaco', monospace;
    }

    /* Aggressive base font application - targets multiple Streamlit containers */
    html, body, [class*="css"],
    .stApp, .main, .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    div, p, span, label {
        font-family: var(--font-sans) !important;
        color: var(--primary-color);
    }

    /* Headers - using Lora serif with stronger specificity */
    h1, h2, h3, h4, h5, h6,
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: var(--font-serif) !important;
        font-weight: 400;
        color: var(--dark-gray);
        letter-spacing: -0.01em;
    }

    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }

    h2 {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* Custom header classes */
    .main-header {
        font-family: var(--font-serif) !important;
        font-size: 2.5rem;
        font-weight: 400;
        color: var(--dark-gray);
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }

    .sub-header {
        font-family: var(--font-sans) !important;
        font-size: 1rem;
        font-weight: 400;
        color: var(--medium-gray);
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Paragraphs and text */
    p, .stMarkdown, [data-testid="stMarkdownContainer"] {
        font-family: var(--font-sans) !important;
    }

    /* Links */
    a {
        font-family: var(--font-sans) !important;
        color: var(--primary-color);
        text-decoration: underline;
        transition: all 0.2s ease;
    }

    a:hover {
        color: var(--light-gray);
        text-decoration: none;
    }

    /* Buttons - target multiple levels */
    .stButton > button,
    button[kind="primary"],
    button[kind="secondary"] {
        font-family: var(--font-sans) !important;
        background-color: var(--primary-color);
        color: white;
        border: 1px solid var(--primary-color);
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 0.02rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--dark-gray) !important;
        border-color: var(--dark-gray) !important;
        color: white !important;
    }

    /* Metrics - cleaner cards */
    [data-testid="stMetric"],
    [data-testid="stMetric"] * {
        font-family: var(--font-sans) !important;
    }

    [data-testid="stMetric"] {
        background-color: white;
        padding: 1.25rem;
        border-radius: 4px;
        box-shadow: 1px 1px 10px rgba(0, 0, 0, 0.1);
        border: none;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] {
        font-family: var(--font-sans) !important;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--medium-gray);
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] {
        font-family: var(--font-sans) !important;
        font-size: 1.75rem;
        font-weight: 500;
        color: var(--dark-gray);
    }

    /* Select boxes and inputs - target labels and values */
    .stSelectbox, .stTextInput, .stNumberInput,
    .stSelectbox *, .stTextInput *, .stNumberInput * {
        font-family: var(--font-sans) !important;
    }

    .stSelectbox label, .stTextInput label, .stNumberInput label,
    [data-baseweb="select"] {
        font-family: var(--font-sans) !important;
        font-weight: 500;
        color: var(--dark-gray);
        font-size: 0.95rem;
    }

    /* Dropdown options */
    [data-baseweb="popover"] *,
    [role="listbox"] *,
    [role="option"] {
        font-family: var(--font-sans) !important;
    }

    /* Code blocks - using Fira Code */
    code, pre, .stCodeBlock,
    [data-testid="stCodeBlock"],
    code *, pre * {
        font-family: var(--font-mono) !important;
        background-color: var(--gray-light);
    }

    code {
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-size: 0.9em;
    }

    pre {
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 1rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        font-family: var(--font-sans) !important;
    }

    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] .element-container h1 {
        font-family: var(--font-sans) !important;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        color: var(--dark-gray);
    }

    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] [role="radiogroup"] label {
        font-family: var(--font-sans) !important;
        font-weight: 500;
        color: var(--dark-gray);
    }

    /* Dataframes and tables */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] *,
    .stDataFrame *,
    table, th, td {
        font-family: var(--font-sans) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color);
        border-radius: 4px;
        overflow: hidden;
    }

    /* AgGrid tables */
    .ag-theme-streamlit,
    .ag-theme-streamlit *,
    .ag-header-cell-text,
    .ag-cell {
        font-family: var(--font-sans) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab-list"] *,
    .stTabs [data-baseweb="tab"] * {
        font-family: var(--font-sans) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: var(--medium-gray);
        border-bottom: 2px solid transparent;
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary-color);
        border-bottom-color: var(--primary-color);
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid var(--border-color);
        margin: 2rem 0;
    }

    /* Expander */
    [data-testid="stExpander"],
    [data-testid="stExpander"] * {
        font-family: var(--font-sans) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border-color);
        border-radius: 4px;
        background-color: white;
    }

    /* Toast notifications */
    .stToast,
    [data-testid="stNotification"],
    [data-testid="stNotification"] * {
        font-family: var(--font-sans) !important;
    }

    /* Footer styling */
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        border-top: 1px solid var(--border-color);
        padding: 1rem 0;
        text-align: center;
        z-index: 999;
        font-family: var(--font-sans);
    }

    .app-footer a {
        color: var(--primary-color);
        text-decoration: underline;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .app-footer a:hover {
        color: var(--light-gray);
        text-decoration: none;
    }

    /* Add padding to main content so footer doesn't overlap */
    .main .block-container {
        padding-bottom: 4rem;
    }
    </style>
""",
        unsafe_allow_html=True,
    )
