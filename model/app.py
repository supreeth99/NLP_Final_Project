import streamlit as st
from model import * 

query = st.text_input('What are you looking for')

pred = intent_detector(query)

st.write(pred)