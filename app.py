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

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Deteksi Gambar", "Tentang Model"])

# Halaman Beranda
if page == "Beranda":
    st.title("Deteksi Kerusakan Bangunan Berbasis Citra")

    st.markdown("""
    Aplikasi ini dikembangkan untuk membantu proses identifikasi kerusakan bangunan pascabencana menggunakan citra digital. Dengan hanya mengunggah gambar bangunan, sistem akan memprediksi apakah bangunan mengalami kerusakan atau tidak.

    ## Tujuan Aplikasi
    - Mempercepat proses inspeksi bangunan terdampak bencana
    - Mengurangi ketergantungan pada inspeksi manual di lapangan
    - Memberikan alternatif evaluasi cepat berbasis AI

    ## Fitur Utama
    - Upload gambar bangunan
    - Deteksi otomatis: rusak atau tidak rusak
    - Hasil prediksi langsung ditampilkan

    ## Cara Menggunakan
    1. Masuk ke halaman Deteksi Gambar
    2. Upload gambar bangunan
    3. Tunggu beberapa detik, hasil akan muncul di layar

    ## Siapa yang Cocok Menggunakan Ini?

    - Mahasiswa teknik sipil atau informatika
    - Relawan kebencanaan
    - Dinas PU atau tim inspeksi
    - Peneliti AI di bidang bangunan & citra digital

    Aplikasi ini cocok digunakan oleh mahasiswa, instansi kebencanaan, serta pihak lain yang memerlukan evaluasi awal kondisi bangunan.
    """)

# Halaman Deteksi Gambar
elif page == "Deteksi Gambar":
    st.title("Deteksi Kerusakan Bangunan")

    uploaded_file = st.file_uploader("Upload Gambar Bangunan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("Mendeteksi..."):
            features = extract_features_from_image(image)
            prediction = model.predict([features])[0]

        if prediction == 1:
            st.success("Hasil Deteksi: Bangunan Rusak")
        else:
            st.success("Hasil Deteksi: Bangunan Tidak Rusak")

# Halaman Tentang Model
elif page == "Tentang Model":
    st.title("Tentang Model Deteksi")

    st.markdown("""
    Model yang digunakan dalam aplikasi ini adalah Random Forest Classifier, yaitu salah satu algoritma machine learning berbasis pohon keputusan yang bekerja secara ansambel (menggabungkan banyak pohon).

    ### Fitur Ekstraksi:
    - Edge Density: Mengukur jumlah tepi/retakan dalam gambar
    - LBP (Local Binary Pattern): Mengidentifikasi tekstur permukaan
    - HOG (Histogram of Oriented Gradients): Menangkap pola arah dan kontur

    ### Evaluasi Model:
    - Akurasi: 75%
    - Presisi kelas rusak: 100%
    - Recall kelas rusak: 60%
    - F1-score: 75%

    Model dilatih menggunakan dataset bangunan rusak dan tidak rusak akibat gempa atau bencana lainnya. Dataset ini kemudian diproses dan diseimbangkan agar hasil klasifikasi tidak bias.

    Aplikasi ini dapat dikembangkan lebih lanjut dengan menambahkan lebih banyak data, menggunakan model deep learning, atau menambahkan klasifikasi tingkat kerusakan (ringan, sedang, berat).
    """)
