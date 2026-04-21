
import streamlit as st
from agent import app

st.title("AI Study Buddy 📚")

user_input = st.text_input("Ask a question")

if user_input:
    result = app.invoke({"question": user_input})
    st.write(result["answer"])
