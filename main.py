import streamlit as st

st.title("Hello World")
with st.form("my_form"):
    name = st.text_input("Enter your name")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.write(f"Hello {name}!")
