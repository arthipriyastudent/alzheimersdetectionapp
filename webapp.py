import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# -------------------------------------------------------------
# Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="Alzheimer’s Disease Classifier", layout="wide")
st.title("🧠 Alzheimer’s Disease Stage Classification System")
st.write("Upload an MRI brain scan to detect the Alzheimer’s stage.")

# -------------------------------------------------------------
# Load the trained model (.h5) — auto-download from Google Drive
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "alzheimers_cnn_model.h5"

    if not os.path.exists(model_path):
        st.info("⬇️ Downloading model from Google Drive...")
        file_id = "1I0SHGmE_GSaLwu35ATnIqNGE7HieY4BT"  # your file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# -------------------------------------------------------------
# Labels and recommendations
# -------------------------------------------------------------
labels = ['Mild Demented', 'Moderate Demented', 'Non-Demented', 'Very Mild Demented']

recommendations = {
    "Non-Demented": {
        "Precautions": [
            "Maintain a healthy diet rich in omega-3 fatty acids.",
            "Engage in regular physical and mental exercises.",
            "Get regular sleep and avoid excessive stress."
        ],
        "Medications": ["No medication needed; maintain brain health."]
    },
    "Very Mild Demented": {
        "Precautions": [
            "Increase mental stimulation (puzzles, reading, memory games).",
            "Maintain social interactions to reduce isolation.",
            "Monitor memory or behavioral changes regularly."
        ],
        "Medications": ["Donepezil (Aricept)", "Rivastigmine (Exelon)"]
    },
    "Mild Demented": {
        "Precautions": [
            "Keep a structured daily routine.",
            "Avoid stressful environments.",
            "Ensure consistent sleep schedule and calm surroundings."
        ],
        "Medications": ["Donepezil", "Galantamine", "Memantine"]
    },
    "Moderate Demented": {
        "Precautions": [
            "24-hour supervision may be required.",
            "Use reminders or visual labels at home.",
            "Consult a neurologist regularly for ongoing care."
        ],
        "Medications": ["Memantine", "Combination therapy (Donepezil + Memantine)"]
    }
}

# -------------------------------------------------------------
# File Upload Section
# -------------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload MRI Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------------
# Prediction and Display
# -------------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🧩 Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict Alzheimer’s Stage"):
        with st.spinner("🧠 Analyzing MRI image..."):
            prediction = model.predict(img_array)
            predicted_label = labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.success(f"✅ **Predicted Stage:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.divider()
        st.subheader("🩺 Precautions")
        for p in recommendations[predicted_label]["Precautions"]:
            st.write(f"- {p}")

        st.subheader("💊 Medications")
        for m in recommendations[predicted_label]["Medications"]:
            st.write(f"- {m}")

        st.divider()
        st.info(
            "**Alzheimer’s disease** causes memory loss and cognitive decline. "
            "Early diagnosis and regular care can help slow progression and improve quality of life."
        )
else:
    st.warning("⚠️ Please upload an MRI image to start diagnosis.")
