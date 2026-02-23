import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import easyocr
import os

# --- Load models ---
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return yolo_model, ocr_reader

yolo_model, ocr_reader = load_models()

# --- Helper functions ---
def yolo_detect_and_crop(image: np.ndarray):
    """
    Run YOLO detection on the image, return image with boxes and cropped plates
    """
    results = yolo_model(image)
    image_with_boxes = image.copy()
    cropped_plates = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            conf = float(box.conf[0]) 
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            if conf < 0.5:
                continue

            # Draw bounding box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Crop detected region
            cropped = image[y1:y2, x1:x2]
            cropped_plates.append(cropped)

    return image_with_boxes, cropped_plates

def ocr_on_crops(crops):
    """
    Run OCR on cropped images, return detected text
    """
    texts = []
    for crop in crops:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = ocr_reader.readtext(crop_rgb)
        for (_, text, prob) in result:
            import re
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            texts.append(clean_text)
    return " ".join(texts)

def np_to_pil(image: np.ndarray):
    """Convert OpenCV image to PIL Image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

def get_image_base64(pil_img):
    """Convert PIL Image to base64"""
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Streamlit UI ---
st.set_page_config(
    page_title="Number Plate Detection",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.stApp {background: linear-gradient(to right, #ff7e5f, #feb47b); color: #333;}
h1, h2, h3 {color: #fff; text-shadow: 2px 2px 4px #000000;}
</style>
""", unsafe_allow_html=True)

st.title("Modern Number Plate Detection")
st.write("Upload an image to detect number plates and extract text using YOLO + OCR.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Number Plate"):
        with st.spinner("Detecting..."):
            # Convert PIL image to OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # YOLO detection + cropping
            image_with_boxes, cropped_plates = yolo_detect_and_crop(image_cv)

            # OCR
            if not cropped_plates:
               st.warning("No number plate detected.")
            else:
               detected_text = ocr_on_crops(cropped_plates)
               st.success(f"Detected Plate Text: **{detected_text}**")

            # Convert result image to PIL
            result_image = np_to_pil(image_with_boxes)
            st.image(result_image, caption="Detected Plates", use_column_width=True)
            st.success(f"Detected Plate Text: **{detected_text}**")

            # Download button
            st.download_button(
                label="Download Processed Image",
                data=base64.b64decode(get_image_base64(result_image)),
                file_name="processed_image.png",
                mime="image/png"
            )
