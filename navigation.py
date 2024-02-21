import streamlit as st
from streamlit.components.v1 import html
from config import SHOW_BUY_ME_A_BEER


change_link_color = """
<style>
    a:link {
    color: green;
    background-color: transparent;
    text-decoration: none;
}
</style>
"""


def make_sidebar():
    """
    Manually create nav/TOC sidebar (default nav sidebar turned off in config.toml)
    """
    with st.sidebar:
        st.markdown(change_link_color, unsafe_allow_html=True)
        st.title("Causal Inference for Data Science")

        st.page_link("Main.py", label="Home", icon="🏠")

        st.write("## Formalizing Causality")

        st.page_link(
            "pages/Motivation.py",
            label="Motivation",
            icon="💡",
        )

        st.page_link(
            "pages/Potential_Outcomes_Framework.py",
            label="Potential Outcomes Framework",
            icon="🪄",
        )

        st.page_link(
            "pages/Graphical_Causal_Models.py",
            label="Graphical Causal Models",
            icon="↔",
        )

        st.write("---")


def make_footer():
    st.write("---")

    # Add buy me a beer button
    if SHOW_BUY_ME_A_BEER:
        bmab_button = """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="dustinstansbury" data-color="#40DCA5" data-emoji="🍻"  data-font="Arial" data-text="Buy me a beer!" data-outline-color="#000000" data-font-color="#ffffff" data-coffee-color="#FFDD00" ></script>
        """

        html(bmab_button, height=70, width=320)
