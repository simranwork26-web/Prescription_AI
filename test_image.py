from models.vlm_model import predict_multiple
from difflib import get_close_matches

# -------- MEDICAL DICTIONARY --------
medical_words = [
    "paracetamol", "ibuprofen", "amoxicillin", "aspirin",
    "cetirizine", "dolo", "azithromycin", "metformin",
    "insulin", "pantoprazole", "omeprazole", "crocin",
    "augmentin", "enzoflam", "pan", "hexigel"
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

# -------- TEST IMAGE --------
image_path = "images/test.jpg"

print("Testing your prescription...")

# OCR
result = predict_multiple(image_path)

# Medical correction
corrected = correct_medical(result)

print("\nRaw Output:")
print(result)

print("\nCorrected Medical Output:")
print(corrected)