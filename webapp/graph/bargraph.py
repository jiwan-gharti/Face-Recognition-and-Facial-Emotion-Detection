import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


def barplot(labels, prediction):
    fig = plt.figure(figsize=(9,7))
    x = labels
    y = prediction * 100
    sns.barplot(x=x, y = y[0])
    st.pyplot(fig)

