import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO



st.set_page_config(page_title="F1 Driver Identifier", layout="wide")


model = tf.keras.models.load_model("model.h5")
class_names = ['carlos_sainz', 'charles_leclerc', 'fernando_alonso', 'lewis_hamilton', 'max_verstappen']

#CSS
page_style = """
<style>
    .stApp {
        background-image: url("https://wallpapercave.com/wp/wp12259035.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }
    .glass-header {
        backdrop-filter: blur(10px);
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem 1rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .glass-header h1 {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
    }
    .glass-header p {
        font-size: 1rem;
        margin-top: 0.5rem;
        color: black;
    }
    .glass-header a {
        color: red;
        text-decoration: none;
        font-weight: 600;
        margin: 0 10px;
    }
    .instruction {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 20px;
        padding: 1rem 1.5rem;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .driver-name {
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(255, 215, 0, 0.85);
        padding: 6px 12px;
        font-weight: bold;
        border-radius: 10px;
        color: black;
        font-size: 1rem;
    }
    .img-wrapper {
        position: relative;
        display: inline-block;
        border-radius: 15px;
        overflow: hidden;
        border: 3px solid #FFD700;
        margin-top: 20px;
    }
</style>
"""

st.markdown(page_style, unsafe_allow_html=True)


st.markdown("""
<div class="glass-header">
    <h1>🏎️ Identify F1 Drivers</h1>
    <p>Developed by <strong>Aditi</strong> | 
    <a href="https://github.com/aditiiprasad" target="_blank">GitHub</a> | 
    <a href="https://www.linkedin.com/in/aditiiprasad" target="_blank">LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)


if 'predicted' not in st.session_state:
    st.session_state.predicted = False

if not st.session_state.predicted:
    st.markdown("""
    <div class="instruction">
        Upload an image of an F1 driver (Charles, Carlos, Max, Lewis, or Fernando) and I'll guess who it is!
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    sample_folder = "sample_images"
    sample_images = [f for f in os.listdir(sample_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if uploaded_file is None:
        st.markdown("### Or select a sample image:")
        cols = st.columns(len(sample_images))
        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(sample_folder, img_name)
            img = Image.open(img_path).resize((250, 250))
            cols[i].image(img, use_container_width=True)
            if cols[i].button(f"Select {img_name}"):
                st.session_state['selected_image'] = img_path
                st.session_state.predicted = True
                st.rerun()
    else:
        img = Image.open(uploaded_file).convert("RGB").resize((250, 250))
        st.session_state['selected_image'] = img
        st.session_state.predicted = True
        st.rerun()

else:
   
    img = st.session_state['selected_image']
    if isinstance(img, str):  # If path
        img = Image.open(img).convert("RGB").resize((250, 250))

   
    img_model = img.resize((100, 100))
    img_array = np.array(img_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)].replace('_', ' ').title()

    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="img-wrapper" style="margin: 0 auto; position: relative; display: inline-block;">
        <div class="driver-name">🏁 {predicted_class}</div>
        <img src="data:image/png;base64,{img_base64}" width="250"/>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
    if st.button("🔄 Try Another Image"):
        st.session_state.predicted = False
        st.session_state.selected_image = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
