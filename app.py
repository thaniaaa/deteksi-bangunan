import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load model
model = joblib.load('model_rf.pkl')

# Fungsi ekstraksi fitur
def extract_features_from_image(image):
    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum() / edges.size

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    hog_feature = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    return np.concatenate([lbp_hist, hog_feature, [edge_density]])

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Deteksi Gambar", "Tentang Model"])

# Halaman Beranda
if page == "Beranda":
    st.title("🧱 Deteksi Kerusakan Bangunan Berbasis Citra")
    st.markdown("""
    Aplikasi ini dikembangkan untuk membantu identifikasi kondisi bangunan pascabencana berdasarkan citra digital.

    ### 📌 Fitur Utama:
    - Upload gambar bangunan
    - Deteksi apakah bangunan **rusak** atau **tidak rusak**
    - Model menggunakan algoritma **Random Forest** dengan fitur **edge**, **LBP**, dan **HOG**
    
    ### 📷 Contoh Penggunaan:
    1. Masuk ke tab *Deteksi Gambar*
    2. Upload foto bangunan
    3. Sistem akan menampilkan hasil klasifikasi

    ---
    """)

# Halaman Deteksi
elif page == "Deteksi Gambar":
    st.title("📤 Deteksi Kerusakan Bangunan")
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("Mendeteksi..."):
            features = extract_features_from_image(image)
            prediction = model.predict([features])[0]
            result = "🟥 Bangunan Rusak" if prediction == 1 else "🟩 Bangunan Tidak Rusak"
        
        st.success(f"Hasil Deteksi: {result}")

# Halaman Tentang Model
elif page == "Tentang Model":
    st.title("📊 Tentang Model")
    st.markdown("""
    ### Algoritma: Random Forest Classifier  
    - Menggunakan 3 fitur utama dari gambar:
      - **Edge Density** (Canny)
      - **Local Binary Pattern (LBP)** untuk tekstur
      - **Histogram of Oriented Gradients (HOG)** untuk bentuk

    ### Evaluasi Model:
    - **Akurasi:** 75%
    - **Presisi kelas rusak:** 1.00
    - **Recall kelas rusak:** 0.60
    - **F1-score:** 0.75

    Model ini cukup baik untuk klasifikasi awal. Performa dapat ditingkatkan dengan dataset lebih besar dan augmentasi data.

    ---
    """)
