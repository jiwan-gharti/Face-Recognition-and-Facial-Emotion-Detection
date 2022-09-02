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


st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)




# =========================================================
# Styling
# =========================================================

# styl = f"""
#     <style>
#         .css-10trblm.e16nr0p30{{
#             padding-top: 10rem;
#             padding-right: 10rem;
#             padding-left: 10rem;
#             padding-bottom: 10rem;
#             background-color: red;
#             color: green;
#         }}
#     </style>
#     """

# st.markdown(styl,unsafe_allow_html=True)




