import streamlit as st
from navigation import make_sidebar, make_footer


make_sidebar()

st.markdown(
    """
#### ⚠️ This app is a WIP ⚠️

An application for teaching (myself, primarily) concepts related to Causal Inference.

"""
)

make_footer()
