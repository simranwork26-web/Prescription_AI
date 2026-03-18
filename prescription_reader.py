from models.vlm_model import predict_multiple
from difflib import SequenceMatcher
import re

# -------- MEDICAL WORD LIST --------
medical_words = [
    "paracetamol", "ibuprofen", "amoxicillin", "aspirin",
    "cetirizine", "dolo", "azithromycin", "metformin",
    "insulin", "pantoprazole", "omeprazole", "crocin",
    "augmentin", "enzoflam", "pan", "hexigel"
]

# -------- FIND BEST MATCH --------
def best_match(word):
    best = word
    best_score = 0

    for med in medical_words:
        score = SequenceMatcher(None, word, med).ratio()

        if score > best_score:
            best_score = score
            best = med

    if best_score > 0.6:
        return best
    else:
        return None

# -------- EXTRACT INFO --------
def extract_info(text):
    words = text.lower().split()

    medicines = []
    dosages = []
    patterns = []

    for w in words:
        clean_word = re.sub(r'[^a-z0-9\-]', '', w)

        # detect dosage (500mg)
        if re.search(r'\d+mg', clean_word):
            dosages.append(clean_word)

        # detect pattern (1-0-1 or similar)
        if re.search(r'\d[-]\d[-]\d', clean_word):
            patterns.append(clean_word)

        # detect medicine
        med = best_match(clean_word)
        if med:
            medicines.append(med)

    return list(set(medicines)), list(set(dosages)), list(set(patterns))

# -------- MAIN --------
image_path = "images/test.jpg"

print("Reading prescription...")

# OCR
raw_text = predict_multiple(image_path)

# Extract structured info
medicines, dosages, patterns = extract_info(raw_text)

print("\n--- RAW OCR OUTPUT ---")
print(raw_text)

print("\n--- DETECTED MEDICINES ---")
print(medicines)

print("\n--- DETECTED DOSAGES ---")
print(dosages)

print("\n--- DOSAGE PATTERN ---")
print(patterns)