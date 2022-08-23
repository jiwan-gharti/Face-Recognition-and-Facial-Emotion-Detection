import streamlit as st
import re

def name_validation():
    name = st.session_state.name
    if len(name) < 2:
        st.error('name field is mandetory and more than 2 character.')
    elif not re.match('^[A-z]+[A-Za-z0-9 _]+',str(name)):
        st.error('name must be start with Character or _ ')



def age_validation():
    age = st.session_state.age
    if not re.match('^[0-9]+$',age):
        st.error('age field should be integer')
    # elif  not isinstance(int(age),int):
    #     print("dfssssssssssssssssssssssssssssssssss")
    #     st.error('age field should be integer')
    elif len(str(age)) < 1:
        st.error('age field is mandetory')
    elif int(age) < 1:
        st.error('age field must be greater than 0')
    elif int(age) > 110:
        st.error('age field should be smaller than 110')

def gender_validation():
    gender = st.session_state.name
    if gender is None:
        st.error('Gender is mandetory')