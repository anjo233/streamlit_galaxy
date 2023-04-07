import streamlit as st
import  time 
from PIL import Image


st.write("计算进度条")

latest = st.empty()
bar = st.progress(0)
for i in range(100):

    latest.text(f'Iteration {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)
"运行结束"


upload_file = st.file_uploader(
    label = "upload image"
)
if upload_file is not None:
    img = Image.open(upload_file)
    img.show()
else:
    st.stop()

