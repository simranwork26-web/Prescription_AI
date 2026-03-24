import pandas as pd
from datasets import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments
)
from PIL import Image
import os

# 🔹 LOAD CSV
df = pd.read_csv("dataset/handwriting_dataset/written_name_train_v2.csv")

# 🔹 FIX COLUMN NAMES
df = df.rename(columns={
    "FILENAME": "image",
    "IDENTITY": "text"
})

# 🔹 CLEAN DATA
df = df.dropna()
df = df[df["text"].str.len() > 0]

# 🔥 USE SMALL SAMPLE FIRST (IMPORTANT)
df = df.sample(5000, random_state=42)

# 🔹 LOAD MODEL
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# 🔥 FINAL CRITICAL FIX (DECODER TOKENS)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# 🔹 CONVERT TO DATASET
dataset = Dataset.from_pandas(df)

# 🔥 DIRECT PREPROCESSING
def load_data(example):
    img_name = example["image"]

    base_path = "dataset/handwriting_dataset/train_v2/train"
    path = os.path.join(base_path, img_name)

    if not os.path.exists(path):
        print("❌ Missing:", img_name)
        return {"pixel_values": None, "labels": None}

    try:
        image = Image.open(path).convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]

        labels = processor.tokenizer(
            example["text"],
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return {"pixel_values": pixel_values, "labels": labels}

    except:
        print("❌ Error:", img_name)
        return {"pixel_values": None, "labels": None}

# 🔹 APPLY PREPROCESS
dataset = dataset.map(load_data, remove_columns=["image", "text"])

# 🔹 REMOVE BAD DATA
dataset = dataset.filter(lambda x: x["pixel_values"] is not None)

# 🔹 TRAINING CONFIG
training_args = TrainingArguments(
    output_dir="./trocr_model",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    fp16=False
)

# 🔹 TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 🔹 TRAIN
trainer.train()

# 🔹 SAVE MODEL
model.save_pretrained("trocr_model")
processor.save_pretrained("trocr_model")

print("✅ Training Complete!")