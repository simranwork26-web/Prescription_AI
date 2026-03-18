from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image, ImageEnhance, ImageOps

print("Loading model...")

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

print("Model loaded!")

# -------- BASIC OCR --------
def predict_text(image):
    image = ImageOps.grayscale(image)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(3.0)

    image = image.convert("RGB")
    image = image.resize((384, 384))

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values, max_new_tokens=100)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text


# -------- MULTI-LINE OCR --------
def predict_multiple(image_path):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    parts = [
        image.crop((0, 0, width, height//5)),
        image.crop((0, height//5, width, 2*height//5)),
        image.crop((0, 2*height//5, width, 3*height//5)),
        image.crop((0, 3*height//5, width, 4*height//5)),
        image.crop((0, 4*height//5, width, height))
    ]

    results = []

    for part in parts:
        text = predict_text(part)
        results.append(text)

    return " ".join(results)
