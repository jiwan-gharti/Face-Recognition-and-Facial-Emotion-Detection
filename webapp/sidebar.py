import streamlit as st


def sidebar():
    with st.sidebar:
        add_selectbox = st.sidebar.selectbox(
            "How would you like to be contacted?",
            ("Email", "Home phone", "Mobile phone")
        )