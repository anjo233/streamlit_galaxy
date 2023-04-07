import streamlit as st
import  time 

st.write("计算进度条")

latest = st.empty()
bar = st.progress(0)
for i in range(100):

    latest.text(f'Iteration {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)
