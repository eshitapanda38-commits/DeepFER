import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="centered"
)

# --------------------------------
# Custom Header
# --------------------------------
st.markdown("""
<h1 style='text-align:center;'>😊 DeepFER</h1>
<h3 style='text-align:center; color:gray;'>
Facial Emotion Recognition using Deep Learning
</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------
# Load Model
# --------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("DeepFER_Model.keras")

model = load_model()

# --------------------------------
# Load Face Detector
# --------------------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# --------------------------------
# Emotion Labels
# --------------------------------
emotion_labels = [
    "😠 Angry",
    "🤢 Disgust",
    "😨 Fear",
    "😊 Happy",
    "😐 Neutral",
    "😢 Sad",
    "😲 Surprise"
]

# --------------------------------
# Upload Image
# --------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Prediction
# --------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    img = np.array(image)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    if len(faces) == 0:
        st.error("❌ No face detected. Please upload a clear front-facing image.")

    else:

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        # -----------------------------
        # Show Images Side by Side
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("😊 Detected Face")
            st.image(face, use_container_width=True)

        st.markdown("---")

        # -----------------------------
        # Preprocess Face
        # -----------------------------
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        # -----------------------------
        # Predict
        # -----------------------------
        with st.spinner("🧠 AI is analyzing the emotion..."):
            prediction = model.predict(face)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        probs = prediction[0]

        # -----------------------------
        # Prediction Result
        # -----------------------------
        st.markdown("## 🧠 Prediction")

        st.success(f"### {emotion_labels[predicted_class]}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.markdown("---")

        # -----------------------------
        # Top 3 Predictions
        # -----------------------------
        st.subheader("🏆 Top 3 Predictions")

        top3 = np.argsort(probs)[::-1][:3]
        medals = ["🥇", "🥈", "🥉"]

        for medal, idx in zip(medals, top3):
            st.write(f"{medal} **{emotion_labels[idx]}** — {probs[idx]*100:.2f}%")

        st.markdown("---")

        # -----------------------------
        # All Probabilities
        # -----------------------------
        st.subheader("📊 All Emotion Probabilities")

        for label, prob in zip(emotion_labels, probs):
            st.progress(
                float(prob),
                text=f"{label} ({prob*100:.2f}%)"
            )
        
       

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")

st.markdown(
    "<center>❤️ Made with Streamlit & TensorFlow by <b>Eshita Panda</b></center>",
    unsafe_allow_html=True
)
