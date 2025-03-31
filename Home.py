import streamlit as st
from langchain.prompts import PromptTemplate

st.write("hello")

a = [1, 2, 3, 4]

b = {"x": 1}

p = PromptTemplate.from_template("xxx")

a

b

st.selectbox("Choose your model.", ["GPT-3", "GPT-4"])
