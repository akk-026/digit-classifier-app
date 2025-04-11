import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('digit_classifier.keras')

# Page Configuration
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="✍️",
    layout="centered",
)

# Sidebar
st.sidebar.title("🧠 Instructions")
st.sidebar.markdown("""
1. Upload a **28x28 handwritten digit image** (png, jpg, jpeg).
2. The app will predict the digit using your trained model.
3. Make sure the digit is centered and clear.
""")

# Main Title
st.markdown("<h1 style='text-align: center;'>🖌️ Digit Classifier</h1>", unsafe_allow_html=True)
st.write("### Upload an image of a handwritten digit:")

# File uploader
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=False, width=200)

    # Preprocess image
    img = img.convert("L").resize((28, 28))  # Grayscale and resize
    img_array = np.array(img)
    img_normalized = img_array / 255.0
    img_input = img_normalized.reshape(1, 28 * 28)

    with st.spinner('🔍 Classifying...'):
        prediction = model.predict(img_input)
        predicted_label = np.argmax(prediction)

    st.success("✅ Classification complete!")
    st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>Predicted Digit: {predicted_label}</h2>", unsafe_allow_html=True)
else:
    st.info("⬆️ Please upload an image to get started.")
