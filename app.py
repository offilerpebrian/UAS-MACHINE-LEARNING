import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Waste Classification", layout="centered")

st.title("🌱 Waste Classification App (MobileNetV2)")
st.write("Upload gambar sampah untuk memprediksi jenisnya.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_model.h5")
    return model

model = load_model()

labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    img = img.resize((160, 160))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        predictions = model.predict(img_array)
        index = np.argmax(predictions)

    st.success(f"Predicted Class: **{labels[index]}**")
    st.write("Confidence:")
    st.bar_chart(predictions[0])
