import string
import streamlit as st
from streamlit.components.v1 import html
from config import SHOW_BUY_ME_A_BEER


def make_toc(sections):
    """Janky generator for adding table-of-contents to the sidebar. Supports
    one level of section nesting.

    Parameters
    ----------
    sections : List[str]
        A list of actual section names on the page to include in the TOC.
        If a section names starts with "-", then we nest it under the previous
        section.
    """
    toc_markdown = ""
    subsection_idx = 0
    subsection_name = list(string.ascii_lowercase)
    n_sections = len(sections)
    for ii, section in enumerate(sections):
        prefix = ""
        postifx = "\\" if ii < n_sections - 1 else ""

        if section.startswith("-"):
            section = section.replace("-", "").strip()
            prefix_name = subsection_name[subsection_idx]
            subsection_idx += 1
            prefix = f"{prefix_name}. "
        else:
            subsection_idx = 0
        section_link = (
            section.lower().replace("(", "").replace(")", "").replace(" ", "-")
        )
        toc_markdown += f"\n{prefix}[{section}](#{section_link}) {postifx}"

    st.sidebar.markdown(toc_markdown)

    # Style the links
    change_toc_link_color = """
    <style>
        a:visited{
            color:None;
            background-color: transparent;
            text-decoration: none;
        }
        a:hover{
            color:red;
            background-color: transparent;
            text-decoration: none;
        }

        a:link {
            color: None;
            background-color: transparent;
            text-decoration: none;
    }
    </style>
    """
    st.sidebar.markdown(change_toc_link_color, True)


def make_sidebar():
    """
    Manually create nav/TOC sidebar (default nav sidebar turned off in config.toml)
    """
    st.set_page_config(layout="wide")

    with st.sidebar:

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
