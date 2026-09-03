import streamlit as st
import cv2
import numpy as np
import PIL

st.title("System Diagnostics")
st.write(f"Streamlit Version: {st.__version__}")
st.write(f"OpenCV Version: {cv2.__version__}")
st.write(f"NumPy Version: {np.__version__}")
st.write("Environment is healthy and running!")
