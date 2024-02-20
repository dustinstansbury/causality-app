import streamlit as st
from streamlit.components.v1 import html
from config import SHOW_BUY_ME_A_BEER


def make_sidebar():
    """
    Manually create sidebar (turned off in config.toml)
    """
    with st.sidebar:
        st.title("Causal Inference Book")

        st.page_link("Main.py", label="Main", icon="🏠")

        st.write("## Basics")
        st.page_link(
            "pages/Potential_Outcomes_Framework.py",
            label="Potential Outcomes Framework",
            icon="🪄",
        )

        st.write("---")


def buy_me_a_beer():
    # Add Buy me a coffee Button at location
    if SHOW_BUY_ME_A_BEER:
        bmab_button = """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="dustinstansbury" data-color="#40DCA5" data-emoji="🍺"  data-font="Arial" data-text="Buy me a beer!" data-outline-color="#000000" data-font-color="#ffffff" data-coffee-color="#FFDD00" ></script>
        """

        return html(bmab_button, height=70, width=320)


def make_footer():
    st.write("---")
    buy_me_a_beer()
