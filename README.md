# Prescription_AI
⚙️ Project Structure

Prescription_AI/

├── app.py
├── main.py
├── easy_prescription.py
├── prescription_reader.py
├── test_image.py
├── README.md
├── requirements.txt
├── .gitignore

├── models/
│   └── vlm_model.py

├── data/
│   └── dataset_loader.py

├── images/
│   └── test.jpg

---

▶️ How to Run

Step 1: Install dependencies

pip install -r requirements.txt

Step 2: Run OCR pipeline

python easy_prescription.py

Step 3: Run main pipeline

python main.py

Step 4: Test with image

python test_image.py

---

📂 Dataset

- Full dataset is not uploaded due to size constraints
- Sample image is available in "/images/test.jpg"
- Dataset loader supports full dataset integration
