import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt



def barplot(labels, prediction):
    fig = plt.figure(figsize=(9,7))
    x = labels
    y = prediction * 100
    sns.barplot(x=x, y = y[0])
    st.pyplot(fig)


# def barplot_function():
#     fig = plt.figure(figsize=(9,7))
#     print('------------------------------')
#     # print(x,y)
#     x = np.array(['real','spoof']),
#     y = [1,0]
#     bar = sns.barplot(x=x, y = y)


#     st.pyplot(fig)

def barplot1(labels,predictions):
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x=labels, y=predictions)
    fig.show()
