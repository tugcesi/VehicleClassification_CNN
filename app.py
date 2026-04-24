import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Vehicle Classifier",
    page_icon="🚗",
    layout="centered"
)

# ── Sabitler ─────────────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)
MODEL_PATH = "src/vehicle_cnn.h5"

CLASS_NAMES = {
    0: "🛺 Baobao",
    1: "🚌 Bus",
    2: "🚗 Car",
    3: "🏍️ Motorcycle",
    4: "🔭 Topdown",
    5: "🛺 Tricycle",
    6: "🚐 Van"
}

# ── Model Yükleme ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

# ── Görsel Ön İşleme ──────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE) / 255.0
    return np.expand_dims(img, axis=0)   # (1, 128, 128, 3)

# ── Arayüz ────────────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Image Classifier")
st.markdown("Bir araç görseli yükle, model hangi araç olduğunu tahmin etsin!")
st.divider()

uploaded_file = st.file_uploader(
    "📁 Görsel Yükle (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📷 Yüklenen Görsel")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 Tahmin Sonuçları")

        with st.spinner("Tahmin yapılıyor..."):
            model     = load_cnn_model()
            img_array = preprocess_image(image)
            preds     = model.predict(img_array, verbose=0)[0]
            top_idx   = int(np.argmax(preds))

        top_label      = CLASS_NAMES[top_idx]
        top_confidence = preds[top_idx] * 100

        st.success(f"**Tahmin: {top_label}**")
        st.metric(label="Güven Skoru", value=f"%{top_confidence:.1f}")

        st.divider()
        st.markdown("**📊 Tüm Sınıf Olasılıkları**")

        for idx in np.argsort(preds)[::-1]:
            label = CLASS_NAMES[idx]
            prob  = preds[idx] * 100
            st.progress(int(prob), text=f"{label}: %{prob:.1f}")

st.divider()
st.caption("Model: Custom CNN (vehicle_cnn.h5) | 7 Sınıf | Input: 128×128")
