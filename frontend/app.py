
import os
import requests
import streamlit as st
from PIL import Image

# -----------------------------------
# Configuration
# -----------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="DiabetesVision",
    page_icon="👁️",
    layout="centered"
)

# -----------------------------------
# Page design
# -----------------------------------

st.title("👁️ DiabetesVision")
st.subheader("Diabetic Retinopathy Image Classification")

st.write(
    "Upload a retinal fundus image to receive an AI-generated "
    "prediction of its diabetic retinopathy severity grade."
)

st.info(
    "Educational and research demonstration only. "
    "This application is not a medical diagnostic tool."
)

# -----------------------------------
# API status
# -----------------------------------

try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.ok:
        status = response.json()

        if status.get("model_loaded"):
            st.success("API connected — model loaded.")
        else:
            st.warning("API connected, but the trained model is not loaded yet.")
    else:
        st.error("API is responding with an error.")
except requests.RequestException:
    st.error("Cannot connect to the API. Please start FastAPI first.")

# -----------------------------------
# Image upload
# -----------------------------------

st.header("Upload retinal image")

uploaded_file = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded retinal image",
        use_container_width=True
    )

    if st.button("Analyze Image", type="primary", use_container_width=True):

        with st.spinner("Analyzing retinal image..."):

            try:
                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                result = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120
                )

                if result.ok:
                    prediction = result.json()

                    st.success("Analysis completed!")

                    st.subheader("Prediction results")

                    st.metric(
                        "Predicted severity grade",
                        f"Grade {prediction['predicted_grade']}"
                    )

                    st.write(
                        "**Classification:**",
                        prediction["prediction"]
                    )

                    st.write(
                        "**Model confidence:**",
                        f"{prediction['confidence'] * 100:.2f}%"
                    )

                    st.subheader("Class probabilities")

                    for grade, probability in prediction["probabilities"].items():
                        st.write(grade)
                        st.progress(float(probability))

                    st.caption(prediction["notice"])

                else:
                    try:
                        detail = result.json().get("detail", result.text)
                    except ValueError:
                        detail = result.text

                    st.error(f"Prediction failed: {detail}")

            except requests.RequestException as error:
                st.error(f"Could not reach the API: {error}")

st.divider()

st.caption(
    "DiabetesVision | Machine Learning Pipeline Project | "
    "African Leadership University"
)
