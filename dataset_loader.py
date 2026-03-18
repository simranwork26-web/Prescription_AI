import os
from PIL import Image

kaggle_path = "data/kaggle_dataset/archive/train_v2/train"

files = os.listdir(kaggle_path)

# filter only image files
image_files = [f for f in files if f.endswith(".png") or f.endswith(".jpg")]

print("Total Images:", len(image_files))

sample = image_files[0]
print("Sample Image:", sample)

img_path = os.path.join(kaggle_path, sample)

img = Image.open(img_path)
img.show()