import streamlit as st
import numpy as np
from PIL import Image
import os

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Classification — CNN",
    page_icon="🚗",
    layout="centered",
)

# ─── Sınıf İsimleri ───────────────────────────────────────────────────────────
# ⚠️ Kendi modelindeki sınıf sırasına göre güncelle!
CLASS_NAMES = [
    "Araba 🚗",
    "Motosiklet 🏍️",
    "Otobüs 🚌",
    "Kamyon 🚚",
    "Bisiklet 🚲",
]

IMG_SIZE = (128, 128)   # Modelinin input boyutuna göre güncelle
DRIVE_LINK = "https://drive.google.com/file/d/1KdaHvkAM_U09jszSmxhDNSG5MWsgDW2H/view?usp=drive_link"

# ─── Model Yükleme ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model("vehicle_cnn.h5")
        return model
    except Exception as e:
        return None

model = load_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ Hakkında")
    st.markdown("""
    Bu uygulama, **CNN (Convolutional Neural Network)** tabanlı bir model kullanarak
    araç görsellerini sınıflandırır.

    **Desteklenen Sınıflar:**
    """)
    for cls in CLASS_NAMES:
        st.markdown(f"- {cls}")

    st.markdown("---")
    st.markdown("**Model:** `vehicle_cnn.h5`")
    st.markdown("**Geliştirici:** [tugcesi](https://github.com/tugcesi)")
    st.markdown(f"**Hugging Face:** [Space](https://huggingface.co/spaces/tugcesi/vehicle_classification_cnn)")

# ─── Ana Başlık ───────────────────────────────────────────────────────────────
st.title("🚗 Vehicle Classification — CNN")
st.markdown("Bir araç fotoğrafı yükle, model hangi araç türü olduğunu tahmin etsin!")
st.markdown("---")

# ─── Model Uyarısı ────────────────────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ **Model dosyası bulunamadı!**\n\n"
        "`vehicle_cnn.h5` dosyasını aşağıdaki Drive bağlantısından indirip "
        "uygulamanın bulunduğu klasöre koy:\n\n"
        f"📥 [Drive'dan İndir]({DRIVE_LINK})"
    )
    st.stop()

# ─── Dosya Yükleme ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Araç fotoğrafı seç (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📸 Yüklenen Görsel")
        st.image(image, use_container_width=True)

    # ─── Tahmin ───────────────────────────────────────────────────────────────
    with col2:
        st.subheader("🔍 Tahmin Sonucu")
        if st.button("Tahmin Et 🚀", use_container_width=True):
            with st.spinner("Model çalışıyor..."):
                img_resized = image.resize(IMG_SIZE)
                img_array = np.array(img_resized, dtype="float32") / 255.0
                img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

                predictions = model.predict(img_array)[0]
                pred_index = int(np.argmax(predictions))
                pred_label = CLASS_NAMES[pred_index]
                confidence = float(predictions[pred_index]) * 100

            st.success(f"### {pred_label}")
            st.metric("Güven Skoru", f"%{confidence:.1f}")
            st.progress(confidence / 100)

            st.markdown("#### 📊 Tüm Sınıf Olasılıkları")
            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                bar_val = float(prob)
                color = "🟩" if i == pred_index else "⬜"
                st.markdown(f"{color} **{cls}**: %{bar_val*100:.1f}")
                st.progress(bar_val)

st.markdown("---")
st.caption("Powered by TensorFlow & Streamlit | [GitHub](https://github.com/tugcesi/VehicleClassification_CNN)")
