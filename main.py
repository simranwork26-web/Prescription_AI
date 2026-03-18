from models.vlm_model import predict_text
import os
import pandas as pd
import re
from difflib import SequenceMatcher, get_close_matches

# -------- CLEAN FUNCTION --------
def clean(text):
    text = text.lower()
    text = text.replace("'", "")
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z]', '', text)
    return text

# -------- SIMILARITY FUNCTION --------
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# -------- MEDICAL DICTIONARY --------
medical_words = [
    "paracetamol", "ibuprofen", "amoxicillin", "aspirin",
    "cetirizine", "dolo", "azithromycin", "metformin",
    "insulin", "pantoprazole", "omeprazole", "crocin",
    "augmentin", "atorvastatin", "amlodipine"
]

# -------- MEDICAL CORRECTION --------
def correct_medical(text):
    words = text.lower().split()
    corrected = []

    for w in words:
        match = get_close_matches(w, medical_words, n=1, cutoff=0.7)
        if match:
            corrected.append(match[0])
        else:
            corrected.append(w)

    return " ".join(corrected)

# -------- PATHS --------
kaggle_path = "data/kaggle_dataset/archive/train_v2/train"
csv_path = "data/kaggle_dataset/archive/written_name_train_v2.csv"

# -------- LOAD CSV --------
df = pd.read_csv(csv_path)

print("Total records in CSV:", len(df))

correct = 0
total = 5

for i in range(total):

    image_name = df.iloc[i, 0]
    actual_text = str(df.iloc[i, 1])

    image_path = os.path.join(kaggle_path, image_name)

    print("\n-----------------------------------")
    print("Image:", image_name)

    # -------- OCR --------
    predicted_text = predict_text(image_path)

    # -------- MEDICAL CORRECTION --------
    corrected_text = correct_medical(predicted_text)

    print("Predicted:", predicted_text)
    print("Corrected:", corrected_text)
    print("Actual   :", actual_text)

    # -------- CLEAN --------
    pred_clean = clean(corrected_text)
    actual_clean = clean(actual_text)

    # -------- SIMILARITY --------
    score = similarity(pred_clean, actual_clean)

    print("Similarity Score:", round(score, 2))

    if score > 0.5:
        correct += 1

# -------- FINAL ACCURACY --------
accuracy = (correct / total) * 100

print("\n===================================")
print("Final Accuracy:", accuracy, "%")