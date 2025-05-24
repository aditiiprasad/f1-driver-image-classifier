import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("model.h5")


class_names = ['carlos_sainz', 'charles_leclerc', 'fernando_alonso', 'lewis_hamilton', 'max_verstappen']

st.title("🏎️ F1 Driver Face Classifier")
st.write("Upload an image of an F1 driver (Charles, Carlos, Max, Lewis, or Fernando) and I'll guess who it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)  

    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"🏁 This looks like **{predicted_class.replace('_', ' ').title()}**!")
