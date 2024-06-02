import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
data1=pd.read_csv('new_model.csv')
data=data1.head()
x=np.array(data1['Wbcc']).reshape(-1,1)
lr=LinearRegression()
lr.fit(x,np.array(data1['Class']))
pie1=['urea','sodium','potasium','hemoglobin','Scandium']
pie2=[57.41 ,137.52,4.63,12.53,3.07]
st.title("ALLERGY PREDICTOR")
nav =st.sidebar.radio("Navigation",["Home","Prediction","About"])
if nav=="Home":
    st.image("homeimg.png",width=300)
    st.write("We have used a well researched data set to predict the severity of ALLERGY")
    if st.button("PAST RECORDS"):
     st.table(data)
    st.write("**CHECK OUT  frequency distributions of components in BLOOD:**")
    plt.subplot(1,1,1)
    plt.title('Distribution of WBCS in blood')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.distplot(data1['Wbcc'],color="blue")
    st.pyplot()
    plt.subplot(2,1,2)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.title('Distribution of RBCS in blood')
    sns.distplot(data1['Rbcc'],color="red")
    st.pyplot()
    plt.subplot(2,2,2)
    plt.figure(figsize=[2,2])
    plt.pie(pie2,labels=pie1,autopct='%0.1f%%')
    st.pyplot()
if nav=="Prediction":
    st.header("Know your allergy on basis of WBC count:")
    st.image("img22.png",width=300)
    st.write("**NOTE: 0 means no allergy and 1 means severe allergy**")
    val=st.number_input("Enter WBC count in your blood sample",0.00,30000.00,step=500.00)
    val=np.array(val).reshape(1,-1)
    pred=lr.predict(val)[0]
    if st.button("predict"):
       st.success(f"your predicted allergy class is {round(pred)} ")
if nav=="About":
    st.title("ABOUT")
    st.write("We try prediction of your allergy class-0 or 1 on basic of our machine learning model and the scientific theorem of increase in wbcs during infection in body.")

