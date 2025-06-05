import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
from tensorflow.keras.models import load_model

# URL model dari Hugging Face Hub
MODEL_URL = "https://huggingface.co/zakialfadilah/best_model_resnet50/resolve/main/best_model_resnet50.keras"
MODEL_PATH = "best_model_resnet50.keras"

# Download model dengan streaming chunk supaya file tidak korup
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    st.write(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")
    model = load_model(MODEL_PATH)
    return model

# Fungsi prediksi gambar
def predict_image(model, image):
    image = image.resize((224, 224))  # Sesuaikan input model
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    return prediction

# Streamlit UI
st.title("ResNet50 Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model (cached)
    model = download_and_load_model()

    # Prediksi gambar
    prediction = predict_image(model, image)

    st.subheader("Prediction Result:")
    st.write(prediction)
else:
    st.info("Please upload an image to get prediction.")
