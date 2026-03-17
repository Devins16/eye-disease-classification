import streamlit as st
from PIL import Image
import numpy as np
import torch
from utils import load_model, predict
from class_explanations import class_explanations

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Eye Disease Classification",
    layout="wide",
)

# ===================== STYLE =====================
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

.title {
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    color: #9ca3af;
    margin-bottom: 20px;
}

img {
    border-radius: 15px;
    object-fit: cover;
}

.badge {
    background-color: #16a34a;
    padding: 5px 10px;
    border-radius: 10px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown('<div class="title">Eye Disease Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based detection using ConvNeXtV2</div>', unsafe_allow_html=True)

# ===================== LOAD MODEL =====================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ===================== LAYOUT =====================
col1, col2 = st.columns([1, 1])

# ===================== LEFT =====================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Upload Eye Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # FIX SIZE (hanya untuk display)
        image_display = image.resize((300, 300))
        st.image(image_display, caption="Input Image")

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== RIGHT =====================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Prediction Result")

    if uploaded_file is not None:
        # ===================== PREDICT =====================
        probs = predict(model, image)

        # ===================== HANDLE SEMUA FORMAT =====================
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        probs = np.array(probs)

        # kalau bentuknya (1, n)
        if probs.ndim > 1:
            probs = probs.flatten()

        # ===================== SAFETY =====================
        if len(probs) == 0:
            st.error("Prediction failed: empty output")
        else:
            class_names = list(class_explanations.keys())

            idx = int(np.argmax(probs))
            confidence = float(np.max(probs)) * 100

            # ===================== RESULT =====================
            st.markdown(f"### {class_names[idx]}")
            st.markdown(
                f'<span class="badge">↑ {confidence:.2f}% confidence</span>',
                unsafe_allow_html=True
            )

            st.write("")
            st.subheader("Probability by Class")

            for i, cls in enumerate(class_names):
                value = float(probs[i]) if i < len(probs) else 0.0
                st.write(cls)
                st.progress(value)

            st.write("")
            st.subheader("Explanation")
            st.info(class_explanations[class_names[idx]])

    else:
        st.info("Upload an image to see prediction")

    st.markdown('</div>', unsafe_allow_html=True)