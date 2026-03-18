import easyocr
import re
from difflib import SequenceMatcher

reader = easyocr.Reader(['en'])

medical_words = [
    "paracetamol", "ibuprofen", "amoxicillin", "aspirin",
    "cetirizine", "dolo", "azithromycin", "metformin",
    "insulin", "pantoprazole", "omeprazole", "crocin",
    "augmentin", "enzoflam", "pan", "hexigel"
]

def best_match(word):
    best = None
    best_score = 0
    for med in medical_words:
        score = SequenceMatcher(None, word.lower(), med).ratio()
        if score > best_score:
            best_score = score
            best = med
    return best if best_score > 0.6 else None

image_path = "images/test.jpg"

print("Reading prescription...")

result = reader.readtext(image_path)
text = " ".join([r[1] for r in result])

print("\n--- RAW OCR OUTPUT ---")
print(text)

words = text.lower().split()

medicines = []
dosages = []
patterns = []
durations = []

for i in range(len(words)):
    w = re.sub(r'[^a-z0-9\-]', '', words[i])

    if w == "":
        continue

    # MEDICINE
    med = best_match(w)
    if med:
        medicines.append(med)

    # DOSAGE
    if i < len(words) - 1:
        next_word = re.sub(r'[^0-9]', '', words[i+1])
        if w.isdigit() and next_word.isdigit():
            combined = w + next_word
            if 100 <= int(combined) <= 1000:
                dosages.append(combined + "mg")

    if re.search(r'\d+mg', w):
        val = re.findall(r'\d+', w)[0]
        if 100 <= int(val) <= 1000:
            dosages.append(val + "mg")

    # FREQUENCY (old)
    if re.search(r'\d[-]\d[-]\d', w):
        patterns.append(w)

    # NEW FREQUENCY DETECTION
    if "daily" in w or "day" in w:
        if i > 0 and words[i-1].isdigit():
            patterns.append(words[i-1] + " times/day")

    if w in ["once", "twice", "thrice"]:
        patterns.append(w + " daily")

    # DURATION
    if i < len(words) - 1:
        next_word = words[i+1]
        if w.isdigit() and ("day" in next_word or "days" in next_word):
            durations.append(w + " days")
        if w.isdigit() and ("week" in next_word or "weeks" in next_word):
            durations.append(w + " weeks")

medicines = list(set(medicines))
dosages = list(set(dosages))
patterns = list(set(patterns))
durations = list(set(durations))

print("\n--- DETECTED MEDICINES ---")
print(medicines)

print("\n--- DETECTED DOSAGE ---")
print(dosages)

print("\n--- FREQUENCY ---")
print(patterns)

print("\n--- DURATION ---")
print(durations)