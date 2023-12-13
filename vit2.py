import torch
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from PIL import Image, ImageStat
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import timm
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
import accelerate

# Load dataset
dataset = load_dataset("trpakov/chest-xray-classification", "full")

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
model.to(device)  # Move model to CUDA device

# Function to preprocess and move data to the device
def transform(examples):
    inputs = feature_extractor(images=examples["image"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to CUDA device
    inputs['labels'] = torch.tensor(examples["labels"]).to(device)  # Move labels to CUDA device
    return inputs

encoded_dataset = dataset.map(transform, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit_model",
    num_train_epochs=5,
    evaluation_strategy="epoch",
    auto_find_batch_size=True

)

# Define metrics computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    return {
        "accuracy": (predictions == labels).astype(np.float32).mean().item(),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }


# Trainer configuration
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model
torch.save(model.state_dict(), 'final.pth')

# Evaluate the model on the test set
evaluation_results = trainer.evaluate(encoded_dataset["test"])

# Display statistics for the test dataset
print("Test Dataset Statistics:")
print(f"Accuracy: {evaluation_results['eval_accuracy']}")
print(f"Precision: {evaluation_results['eval_precision']}")
print(f"Recall: {evaluation_results['eval_recall']}")
print(f"F1 Score: {evaluation_results['eval_f1']}")

training_history = trainer.state.log_history

# Extract accuracy and loss values
epochs = []
train_loss = []
eval_loss = []
train_accuracy = []
eval_accuracy = []

for entry in training_history:
    if 'loss' in entry:
        epochs.append(entry['epoch'])
        train_loss.append(entry['loss'])
    if 'eval_loss' in entry:
        eval_loss.append(entry['eval_loss'])
    if 'eval_accuracy' in entry:
        eval_accuracy.append(entry['eval_accuracy'])

# Plotting
plt.figure(figsize=(12, 5))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, eval_loss, label='Validation Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, eval_accuracy, label='Validation Accuracy')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
#save plot 
plt.savefig('plot.png')