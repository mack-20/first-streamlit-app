import streamlit as st
import os

st.title("train.py")

current_dir = os.path.dirname(__file__)
train_path = os.path.join(current_dir, '..', 'app.py')
train_path = os.path.abspath(train_path)

try: 
    with open(train_path, "r") as file:
        file_content = file.read()
    st.code(file_content, language="python")
except FileNotFoundError:
    st.error(f"Error: The file '{train_path}' was not found")
except Exception as e:
    st.error(f"An error occured: {e}")