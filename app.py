# app.py - Fruit Classifier Streamlit App (MobileNetV2)

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps
import io
import pandas as pd

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# SETTINGS (MUST MATCH TRAINING)
# ===========================

IMG_SIZE = (160, 160)  # must match the model training input

# IMPORTANT:
# The order of CLASS_NAMES MUST match the training folder order used by your dataset loader.
# In tf.keras.utils.image_dataset_from_directory, class_names are typically alphabetical by folder name.
CLASS_NAMES = [
    'Apple',
    'Banana',
    'avocado',
    'cherry',
    'kiwi',
    'mango',
    'orange',
    'pineapple',
    'strawberries',
    'watermelon'
]

FRUIT_INFO = {
    'Apple': "🍎 Rich in fiber and vitamin C. Usually comes in red and green varieties.",
    'Banana': "🍌 Great source of potassium. Perfect pre-workout snack!",
    'avocado': "🥑 Packed with healthy fats and nutrients. Perfect for guacamole!",
    'cherry': "🍒 Sweet and tart! Rich in antioxidants and anti-inflammatory compounds.",
    'kiwi': "🥝 Packed with vitamin C - more than oranges!",
    'mango': "🥭 Known as the king of fruits. Rich in vitamins A and C.",
    'orange': "🍊 Famous for vitamin C. Great for boosting immunity.",
    'pineapple': "🍍 Contains bromelain, an enzyme that aids digestion.",
    'strawberries': "🍓 Loaded with vitamin C and antioxidants. Great for heart health!",
    'watermelon': "🍉 92% water! Perfect for staying hydrated in summer."
}

MODEL_PATH = "student_mobilenetv2_transfer_learning.keras"

# ===========================
# LOAD MODEL
# ===========================

@st.cache_resource
def load_trained_model():
    with st.spinner("Loading AI model..."):
        return load_model(MODEL_PATH)

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===========================
# PREDICTION
# ===========================

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Prepare image for MobileNetV2:
    - resize to IMG_SIZE
    - ensure RGB
    - convert to float32
    - apply mobilenet_v2.preprocess_input (scales to expected range)
    - add batch dimension
    """
    try:
        img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    except AttributeError:
        img = ImageOps.fit(img, IMG_SIZE, Image.LANCZOS)

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img).astype(np.float32)  # shape (H, W, 3)
    arr = preprocess_input(arr)               # MobileNetV2-specific scaling
    arr = np.expand_dims(arr, axis=0)         # shape (1, H, W, 3)
    return arr

def predict_fruit(img: Image.Image):
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]  # shape (num_classes,)

    idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100.0

    all_predictions = [
        {"fruit": CLASS_NAMES[i], "probability": float(preds[i]) * 100.0}
        for i in range(len(CLASS_NAMES))
    ]
    all_predictions.sort(key=lambda d: d["probability"], reverse=True)

    return predicted_class, confidence, all_predictions

# ===========================
# UI
# ===========================

st.title("🍎🍌🍊 Fruit Classifier")
st.markdown("""
Upload a fruit image and let AI identify it!  
**Supported fruits:** Apple, Banana, Avocado, Cherry, Kiwi, Mango, Orange, Pineapple, Strawberries, Watermelon
""")

st.divider()

uploaded_file = st.file_uploader(
    "Choose a fruit image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image of a fruit"
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(image_bytes))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(img, use_container_width=True)

        with col2:
            st.subheader("🤖 AI Prediction")

            with st.spinner("Analyzing fruit..."):
                predicted_fruit, confidence, all_predictions = predict_fruit(img)

            st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <h2 style='color: #2e7d32; margin: 0;'>{predicted_fruit}</h2>
                    <p style='color: #558b2f; font-size: 20px; margin: 5px 0;'>
                        Confidence: {confidence:.2f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence / 100.0))

            if predicted_fruit in FRUIT_INFO:
                st.info(FRUIT_INFO[predicted_fruit])

        st.divider()
        st.subheader("📊 All Fruit Probabilities")

        df = pd.DataFrame(all_predictions)
        df["probability"] = df["probability"].map(lambda x: f"{x:.2f}%")
        df.columns = ["Fruit", "Probability"]
        df.index = range(1, len(df) + 1)
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Probability Distribution")
        chart_data = pd.DataFrame({
            "Fruit": [p["fruit"] for p in all_predictions],
            "Probability (%)": [p["probability"] for p in all_predictions],
        })
        st.bar_chart(chart_data.set_index("Fruit"))

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.error("Please make sure you uploaded a valid image file.")
else:
    st.info("👆 Please upload a fruit image to get started!")
    st.markdown("""
    ### How to use:
    1. Click 'Browse files' or drag and drop an image
    2. Wait for AI to analyze it
    3. View the prediction and confidence scores
    4. Explore all fruit probabilities

    ### Tips for best results:
    - Use clear, well-lit images
    - Make sure the fruit is visible
    - One fruit at a time works best
    """)

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses **Transfer Learning** with **MobileNetV2**
    to classify fruit images.

    **Model:** MobileNetV2 (pre-trained on ImageNet)  
    **Framework:** TensorFlow/Keras  
    **Interface:** Streamlit
    """)
