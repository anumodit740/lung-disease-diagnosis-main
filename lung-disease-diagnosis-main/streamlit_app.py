import os
import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

from xray.ml.model.arch import Net


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="X-ray Pneumonia Detection",
    layout="centered"
)

st.title("X-ray Pneumonia Detection")
st.write("Upload a chest X-ray image to classify it as Normal or Pneumonia.")


# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xray_model.pth")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


model = load_model()


# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


label_map = {0: "Normal", 1: "Pneumonia"}


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

    if st.button("Predict"):

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, dim=1)

        prediction = label_map[pred.item()]
        confidence_score = confidence.item() * 100

        st.subheader("Prediction Result")
        st.write(f"Prediction: **{prediction}**")
        st.write(f"Confidence: {confidence_score:.2f}%")
