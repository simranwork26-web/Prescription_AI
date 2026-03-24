import streamlit as st
from PIL import Image
import torch
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import cv2
import numpy as np
from rapidfuzz import process

from medicine_db import medicine_list

# -------------------------------
# LOAD MODELS
# -------------------------------

@st.cache_resource
def load_models():
    processor = TrOCRProcessor.from_pretrained("trocr_model")
    model = VisionEncoderDecoderModel.from_pretrained("trocr_model")
    easy_reader = easyocr.Reader(['en'])
    return processor, model, easy_reader

processor, model, easy_reader = load_models()

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------

def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh

# -------------------------------
# CLEAN TEXT
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# -------------------------------
# OCR
# -------------------------------

def extract_text(image):

    processed = preprocess_image(image)
    pil_img = Image.fromarray(processed).convert("RGB")

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    easy_text = " ".join(easy_reader.readtext(np.array(image), detail=0))

    return clean_text(trocr_text + " " + easy_text)

# -------------------------------
# PRIORITY MEDICINES (IMPORTANT)
# -------------------------------

priority_medicines = [
    # Psychiatric / Neuro (your current domain)
    "sizodon plus",
    "sizodon",
    "ativan",
    "rivotril",
    "clonazepam",
    "sertraline",
    "quetiapine",
    "olanzapine",
    "risperidone",
    "escitalopram",
    "fluoxetine",
    "paroxetine",
    "venlafaxine",
    "duloxetine",

    # Common general medicines
    "paracetamol",
    "crocin",
    "calpol",
    "dolo",
    "ibuprofen",
    "combiflam",
    "aspirin",

    # Antibiotics
    "amoxicillin",
    "azithromycin",
    "ciprofloxacin",
    "cefixime",
    "doxycycline",

    # Gastro / acidity
    "pantoprazole",
    "omeprazole",
    "ranitidine",
    "esomeprazole",

    # Diabetes
    "metformin",
    "glimepiride",
    "insulin",

    # Blood pressure
    "amlodipine",
    "losartan",
    "telmisartan",
    "atenolol",
    "metoprolol",

    # Vitamins / supplements
    "becosules",
    "neurobion",
    "vitamin d",
    "calcium",

    # Others commonly seen
    "montair fx",
    "levocetirizine",
    "cetirizine",
    "ondansetron",
    "domperidone",
    "rabeprazole"
]

# -------------------------------
# MATCH FUNCTION (SMART)
# -------------------------------

def match_medicine(word):

    word = word.lower()

    # Priority match first
    for med in priority_medicines:
        if word in med or med in word:
            return med.title(), 95

    # Fallback fuzzy match
    if len(word) < 4:
        return None, 0

    result = process.extractOne(
        word,
        MEDICINES,
        score_cutoff=85
    )

    if result:
        return result[0].title(), result[1]

    return None, 0

# -------------------------------
# DOSAGE
# -------------------------------

def extract_dosage(text):
    match = re.findall(r"\b\d+\s?(mg|ml)?\b", text)
    return match[0] if match else ""

# -------------------------------
# FREQUENCY
# -------------------------------

def extract_frequency(text):
    patterns = ["once", "twice", "daily", "1 0 1", "0 1 0"]

    for p in patterns:
        if p in text:
            return p
    return ""

# -------------------------------
# MAIN EXTRACTION (CONTEXT BASED)
# -------------------------------

def extract_data(text):

    words = text.split()
    results = []
    seen = set()

    for i in range(len(words)):

        # Only look near prescription keywords
        if words[i] in ["tab", "tablet", "cap", "capsule"]:

            for j in range(1, 4):  # next 3 words
                if i + j < len(words):

                    phrase = words[i + j]

                    med, score = match_medicine(phrase)

                    if med and med not in seen:
                        seen.add(med)

                        results.append({
                            "medicine": med,
                            "confidence": round(score, 2),
                            "dosage": extract_dosage(text),
                            "frequency": extract_frequency(text)
                        })

    return results[:6]  # limit to realistic count

# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("💊 Prescription Reader (Context-Aware AI)")

uploaded_file = st.file_uploader("Upload Prescription", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Read Prescription"):

        st.success("Extraction Complete")

        text = extract_text(image)

        st.subheader("🧾 Raw Text")
        st.write(text)

        data = extract_data(text)

        st.subheader("💊 Final Prescription Data")

        if data:
            st.dataframe(data)
        else:
            st.warning("No medicines detected")