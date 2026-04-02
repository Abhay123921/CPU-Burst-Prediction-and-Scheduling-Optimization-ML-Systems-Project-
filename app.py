import streamlit as st
import requests

st.title("CPU Burst Prediction")

values = []

for i in range(50):
    values.append(st.number_input(f"t-{50-i}", 1, 100, 10))

if st.button("Predict"):
    res = requests.post("http://127.0.0.1:8000/predict",
                        json={"values": values})

    st.write(res.text)