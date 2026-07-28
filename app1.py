import numpy as np
import pandas  as pd
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

minMax=joblib.load("minMax.joblib")
x_enc=joblib.load("x_enc.joblib")
x_enc=list(dict.fromkeys(x_enc))
model=load_model("batch2_deep.keras")

st.set_page_config(
    page_title="deep learning project batch 1",
    page_icon="👍",
    layout="wide"
)

st.header("THIS THE PROJECT TO PREDICT SALARY")
st.write("IN THIS PROJECT WE USED DEEP LEARNING CONCEPT")
st.subheader("THE MODEL IS TRAINED BY 25 EPOCHS")

age=st.number_input("AGE")
experience=st.number_input("Experience",0,40,5)
gender=st.selectbox("Gender",["MALE","FEMALE"])
education=st.selectbox("Education",["Bachelors","Diploma","High School","Masters","PhD"])
country=st.selectbox("Country",["USA","UK","India","Canada","Australia","Germany"])

if st.button("Predict"):
    sample=pd.DataFrame({
        "Age":[age],
        "Experience":[experience],
        "Gender":[gender],
        "Education":[education],
        "Country":[country]
    })

    sample=pd.get_dummies(sample,columns=["Gender","Education","Country"])

    sample=sample.reindex(columns=x_enc,fill_value=0)

    sample=minMax.transform(sample)

    prediction=model.predict(sample,verbose=0)

    st.success(f"PREDICTED SALARY IS : /-{prediction[0][0]:,.2f}")