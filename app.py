
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
import joblib

model = joblib.load('model_rf.pkl')

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

st.title("🧱 Deteksi Kerusakan Bangunan")
st.write("Upload gambar bangunan dan sistem akan mendeteksi apakah rusak atau tidak.")

uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    with st.spinner("Mendeteksi..."):
        features = extract_features_from_image(image)
        prediction = model.predict([features])[0]
        result = "🟥 Bangunan Rusak" if prediction == 1 else "🟩 Bangunan Tidak Rusak"

    st.success(f"Hasil Deteksi: {result}")
