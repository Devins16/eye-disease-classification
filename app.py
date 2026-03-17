import streamlit as st
from PIL import Image
from utils import load_model, predict
from class_explanations import class_explanations

st.set_page_config(
    page_title="Eye Disease AI",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
text-align:center;
margin-bottom:10px;
}

.sub-title{
text-align:center;
color:gray;
margin-bottom:40px;
}

.card{
padding:20px;
border-radius:15px;
background-color:#ffffff;
box-shadow:0 4px 12px rgba(0,0,0,0.08);
margin-bottom:20px;
}

.result-card{
padding:15px;
border-radius:12px;
background-color:#f8f9fa;
margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown(
'<div class="main-title">Eye Disease Classification</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="sub-title">AI-based detection of eye diseases using ConvNeXtV2</div>',
unsafe_allow_html=True
)

model = load_model()

# ---------- LAYOUT ----------
col1, col2 = st.columns([1,1])

# ---------- UPLOAD ----------
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Upload Eye Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- PREDICTION ----------
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Prediction Result")

    if uploaded_file:

        results = predict(model, image)

        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1], reverse=True)
        )

        top_class = list(sorted_results.keys())[0]
        top_score = list(sorted_results.values())[0]

        st.metric(
            label="Predicted Disease",
            value=top_class.capitalize(),
            delta=f"{top_score*100:.2f}% confidence"
        )

        st.write("### Probability by Class")

        for cls, prob in sorted_results.items():

            percent = prob * 100
            
            col_label, col_bar, col_percent = st.columns([2,6,1])

            with col_label:
                st.write(cls.capitalize())

            with col_bar:
                st.progress(float(prob))

            with col_percent:
                st.write(f"{percent:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- EXPLANATION ----------
if uploaded_file:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Medical Explanation")

    st.write(class_explanations[top_class])

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- FOOTER ----------
st.markdown(
"""
<br><br>
<center style="color:gray">
Eye Disease Classification Model • ConvNeXtV2 Architecture
</center>
""",
unsafe_allow_html=True
)