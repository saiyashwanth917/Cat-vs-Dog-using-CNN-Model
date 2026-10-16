
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱 Cats vs Dogs Classifier")
st.write("Upload an image to classify it using a custom CNN model.")

# Model file location
MODEL_PATH = (
    Path(__file__).parent / "cats_vs_dogs_cnn_Model .keras"
)

# Load the trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

if not MODEL_PATH.exists():
    st.error("Model file not found!")
    st.write("Expected filename: cats_vs_dogs_cnn_Model .keras")
    st.write("Place the model file in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
    st.success("CNN model loaded successfully!")
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Read and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=350
        )

        # Predict button
        if st.button("Predict", type="primary"):
            # Preprocess image exactly as during training
            resized_image = image.resize((160, 160))

            image_array = np.array(
                resized_image,
                dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Get prediction from the custom CNN
            with st.spinner("Predicting..."):
                prediction = float(
                    model.predict(
                        image_array,
                        verbose=0
                    )[0][0]
                )

            # Class mapping: 0 = Cat, 1 = Dog
            if prediction >= 0.5:
                label = "DOG 🐶"
                confidence = prediction
            else:
                label = "CAT 🐱"
                confidence = 1.0 - prediction

            # Display results
            st.subheader(f"Prediction: {label}")
            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                "The prediction is based on your trained CNN model. "
                "Confidence does not guarantee correctness."
            )

    except Exception as e:
        st.error(f"Error processing image: {e}")
