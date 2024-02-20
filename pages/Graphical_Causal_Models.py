import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns


from navigation import make_sidebar, make_footer


make_sidebar()

SHOW_CODE = st.sidebar.toggle("Hide/Show Python Code", value=True)

# Sidebar TOC
st.sidebar.markdown(
    """

"""
)

"""
# Graphical Causal Models
---

"""


make_footer()
