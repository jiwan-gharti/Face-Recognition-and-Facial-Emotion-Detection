import base64
import streamlit as st



st.set_page_config(page_title = 'Computer Vision', page_icon='static/AI vision.png',layout='wide')
st.title("Face And Facial Expression Recognition")
st.sidebar.header('Sidebar')




def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./static/home.png') 



