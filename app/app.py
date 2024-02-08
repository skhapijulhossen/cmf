import streamlit as st
import time as ti

## OUTPUT VARIABLE
test = False

st.header(':red[Image Forgery Detection] :mag: ', divider = 'blue' )

st.file_uploader("Upload your Image")


if st.button("Detect"):
    with st.spinner("Analyzing"):
        ti.sleep(3)
    if(test == True):
        st.header("Forged")
       ##image output
        st.image("demo.png")
    else:
        st.header("Not Forged")

    

