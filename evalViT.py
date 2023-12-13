from transformers import ViTFeatureExtractor, Trainer, ViTForImageClassification
from datasets import load_dataset
import numpy as np
from PIL import Image
import torch
import os

misclassified_images_folder = "misclassified_images"

if not os.path.exists(misclassified_images_folder):
    os.makedirs(misclassified_images_folder)
dataset = load_dataset("trpakov/chest-xray-classification", "full")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform(examples):
    return feature_extractor(examples["image"], return_tensors="pt")

encoded_dataset = dataset.map(transform, batched=True)

model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)

model_path = 'final.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

trainer = Trainer(model=model)
predictions = trainer.predict(encoded_dataset["test"])

pred_labels = np.argmax(predictions.predictions, axis=-1)
actual_labels = predictions.label_ids
misclassified_indices = np.where(pred_labels != actual_labels)[0]

for idx in misclassified_indices:
    idx = int(idx)  # Convert NumPy int64 to native Python int

    # Actual and predicted labels for the misclassified image
    actual_label = actual_labels[idx]
    predicted_label = pred_labels[idx]

    # Folder for this type of misclassification
    misclassification_type_folder = f"misclassified_images/actual_{actual_label}_predicted_{predicted_label}"

    # Create the folder if it doesn't exist
    if not os.path.exists(misclassification_type_folder):
        os.makedirs(misclassification_type_folder)

    # Directly access the image object
    image = dataset["test"][idx]["image"]

    # Save the image in the corresponding folder
    image_path = os.path.join(misclassification_type_folder, f"{idx}_misclassified.png")
    image.save(image_path)

print("Misclassified images saved in separate folders.")