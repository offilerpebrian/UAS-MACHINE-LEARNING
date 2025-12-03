import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Waste Classification", layout="centered")

st.title("♻ Waste Classification App (TFLite Version)")

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="waste_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB").resize((160, 160))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    index = np.argmax(predictions)
    st.success(f"Predicted Class: **{labels[index]}**")
    st.write("Confidence:")
    st.bar_chart(predictions)
