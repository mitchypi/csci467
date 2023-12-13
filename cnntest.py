import torch
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset, Image, Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from PIL import Image, ImageStat
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image


dataset = load_dataset("trpakov/chest-xray-classification", "full")

NUM_CLASSES = 2
NUM_EPOCHS = 10

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform(batch):
    images = [preprocess(item['image']) for item in batch]
    labels = [item['labels'] for item in batch]

    return torch.stack(images), torch.tensor(labels)


train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=transform)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



loss_per_epoch = []
validation_accuracy_per_epoch = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_per_epoch.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_accuracy = 100 * correct / total
    validation_accuracy_per_epoch.append(epoch_accuracy)
    print(f'Validation Accuracy: {epoch_accuracy}%')

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return precision, recall, f1

def test_and_save_misclassified(model, test_loader, device, save_dir="misclassified_images"):
    model.eval()
    all_labels = []
    all_predictions = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for i, (label, pred) in enumerate(zip(labels, predicted)):
                if label != pred:
                    misclassified_subdir = os.path.join(save_dir, f"actual_{label.item()}_predicted_{pred.item()}")
                    if not os.path.exists(misclassified_subdir):
                        os.makedirs(misclassified_subdir)

                    img_path = os.path.join(misclassified_subdir, f"misclassified_{i}.png")
                    save_image(images[i].cpu(), img_path)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = (all_predictions == all_labels).mean()
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = test_and_save_misclassified(model, test_loader, device)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

                 