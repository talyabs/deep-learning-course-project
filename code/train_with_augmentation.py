import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer, DefaultDataCollator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Define paths
train_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
test_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Test"

def list_files(path):
    print(os.listdir(path))
    for folder in os.listdir(path):
        if folder == '.DS_Store':
            continue
        print(folder)
        for subfolder in os.listdir(path + '/' + folder):
            print(subfolder)

# Custom Dataset class
class SkinCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(root_dir))) if label != '.DS_Store'}
        
        for label in self.label_map:
            for img_name in os.listdir(os.path.join(root_dir, label)):
                self.images.append(os.path.join(root_dir, label, img_name))
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)]),  # Add Gaussian Blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the same transform to the test dataset but without augmentations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SkinCancerDataset(root_dir=train_path, transform=transform)
test_dataset = SkinCancerDataset(root_dir=test_path, transform=test_transform)

# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(train_dataset.label_map)  # Number of classes in your dataset
)

# Preprocess function
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"  # Use the correct argument
)

# Custom trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


def preprocess_function(examples):
    inputs = feature_extractor([Image.open(x).convert("RGB") for x in examples['image']], return_tensors="pt")
    inputs['labels'] = examples['label']
    return inputs

data_collator = DefaultDataCollator()

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image, label = example["pixel_values"], example["labels"]
        image = feature_extractor(images=image, return_tensors="pt").pixel_values
        return {"pixel_values": image.squeeze(), "labels": label}
    
train_dataset_hf = CustomDataset(train_dataset)
test_dataset_hf = CustomDataset(test_dataset)

# Instantiate the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=test_dataset_hf,
    data_collator=data_collator,
    tokenizer=feature_extractor,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Function to reverse normalization
def reverse_normalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    return tensor * std + mean

# Generate detailed evaluation report
def evaluate_model(trainer, test_dataset_hf):
    # Get predictions
    predictions = trainer.predict(test_dataset_hf)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Classification report
    report = classification_report(labels, preds, target_names=[key for key in train_dataset.label_map])
    print(report)

    # Save classification report to file
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Save some example predictions
    correct_preds = np.where(preds == labels)[0]
    incorrect_preds = np.where(preds != labels)[0]

    def save_examples(indices, prefix):
        for i in indices[:10]:  # Save first 10 examples
            img_path = test_dataset.images[i]
            img = Image.open(img_path)
            img_tensor = transform(img).unsqueeze(0)
            img_tensor = reverse_normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = img_tensor.squeeze().permute(1, 2, 0).numpy()
            plt.imshow(img_tensor)
            plt.title(f"True: {labels[i]}, Pred: {preds[i]}")
            plt.savefig(f"{prefix}_example_{i}.png")
            plt.close()

    save_examples(correct_preds, "correct")
    save_examples(incorrect_preds, "incorrect")

evaluate_model(trainer, test_dataset_hf)
