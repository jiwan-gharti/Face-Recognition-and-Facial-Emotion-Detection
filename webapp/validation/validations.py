import streamlit as st

def name_validation():
    name = st.session_state.name
    if len(name) < 2:
        st.error('name field is mandetory and more than 2 character.')


def age_validation():
    age = st.session_state.age
    if len(age) < 1:
        st.error('age field is mandetory')

def gender_validation():
    gender = st.session_state.name
    if gender is None:
        st.error('Gender is mandetory')