import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# App config (for mobile optimization)
st.set_page_config(
    page_title="Guava Leaf Detector 🌿",
    page_icon="🌿",
    layout="wide",  # Full-width layout for mobile
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Set Background Image using CSS (Nature-themed)
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1524592094714-0f065b1f2378?ixlib=rb-4.0.3&auto=format&fit=crop&w=1650&q=80");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Title and Description
st.markdown("<h1 style='text-align: center; font-size:36px; color: #2E8B57;'>🌿 Guava Leaf Disease Detection 🌿</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-size:20px; color: #556B2F;'>Upload a leaf image and we'll detect the disease instantly! 📱</h3>", unsafe_allow_html=True)

# Load Model
model = tf.keras.models.load_model('b_m-net_guava_leaf_model.h5')

# Prediction Classes
CLASS_NAMES = ['Caterpillars', 'Cutting Weevil', 'Die Back', 'Healthy', 'Mealybug Pests', 'Red Rust', 'Yellow spot']  # Update with your classes

# Prediction Function
def import_and_predict(image_data, model):
    image = Image.open(image_data).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

# File Uploader
uploaded_file = st.file_uploader("Choose a guava leaf image...", type=["jpg", "jpeg", "png"])

# When file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🌿 Uploaded Leaf Image', width=300)

    with st.spinner('🌀 Analyzing the leaf... Please wait...'):
        time.sleep(2)  # Optional delay for smoother experience
        preds = import_and_predict(uploaded_file, model)
        predicted_class = CLASS_NAMES[np.argmax(preds)]

    st.success(f"✅ Prediction: **{predicted_class}**")
