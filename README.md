# 🚗 Vehicle Classification with CNN

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![License](https://img.shields.io/github/license/tugcesi/VehicleClassification_CNN)

CNN tabanlı bir derin öğrenme modeli kullanarak araç görsellerini sınıflandıran bir proje.
Canlı demo için **Hugging Face Space**'i ziyaret edebilirsin 👇

🤗 **[Hugging Face Demo](https://huggingface.co/spaces/tugcesi/vehicle_classification_cnn)**

---

## 📌 Proje Özeti

Bu projede araç görsellerini otomatik olarak sınıflandıran bir **CNN (Convolutional Neural Network)** modeli geliştirilmiştir. Model, farklı araç türlerini yüksek doğrulukla ayırt edebilmektedir.

**Desteklenen Sınıflar:**
- 🚗 Araba
- 🏍️ Motosiklet
- 🚌 Otobüs
- 🚚 Kamyon
- 🚲 Bisiklet

---

## 🏗️ Model Mimarisi

| Katman | Detay |
|--------|-------|
| Input | 128 × 128 × 3 (RGB) |
| Conv2D Bloklar | Birden fazla konvolüsyon + BatchNorm + MaxPooling |
| Dropout | Overfitting önleme |
| Dense | Tam bağlantılı katmanlar |
| Output | Softmax — çoklu sınıf tahmini |

---

## 🚀 Kurulum & Çalıştırma

### 1. Repoyu Klonla
```bash
git clone https://github.com/tugcesi/VehicleClassification_CNN.git
cd VehicleClassification_CNN
```

### 2. Gereksinimleri Yükle
```bash
pip install -r requirements.txt
```

### 3. Modeli İndir
Model dosyası (25 MB+) büyük olduğundan GitHub'a eklenememiştir.
Aşağıdaki Drive bağlantısından `vehicle_cnn.h5` dosyasını indirip **repo kök dizinine** koy:

📥 **[Google Drive — vehicle_cnn.h5](https://drive.google.com/file/d/1KdaHvkAM_U09jszSmxhDNSG5MWsgDW2H/view?usp=drive_link)**

### 4. Uygulamayı Başlat
```bash
streamlit run app.py
```

Tarayıcında `http://localhost:8501` adresini aç.

---

## 📁 Dosya Yapısı

```
VehicleClassification_CNN/
├── app.py                                  # Streamlit uygulaması
├── vehicle_cnn.h5                          # Model (Drive'dan indir)
├── vehicle-classification-using-cnn.ipynb # Eğitim notebook'u
├── requirements.txt                        # Python gereksinimleri
├── .gitignore
└── README.md
```

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım |
|-----------|----------|
| Python | Temel dil |
| TensorFlow / Keras | Model eğitimi & tahmin |
| Streamlit | Web arayüzü |
| Pillow | Görsel işleme |
| NumPy | Sayısal işlemler |

---

## 📓 Notebook

`vehicle-classification-using-cnn.ipynb` dosyasında şunlar yer almaktadır:
- Veri yükleme ve ön işleme
- CNN model mimarisinin oluşturulması
- Eğitim süreci (callbacks, augmentation)
- Performans görselleştirme (loss, accuracy, confusion matrix)
- Model kaydı

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
