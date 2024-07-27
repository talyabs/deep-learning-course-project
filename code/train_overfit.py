import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from PIL import Image
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer, DefaultDataCollator
from transformers import TrainerCallback
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Define paths
train_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
test_path = "/data/talya/deep-learning-course-project/Skin cancer ISIC The International Skin Imaging Collaboration/Test"

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

# Define transformations without data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use a small subset of the training data
train_dataset = SkinCancerDataset(root_dir=train_path, transform=transform)
small_train_dataset = Subset(train_dataset, range(50))  # Use only 10 samples

# Use the same transform for the test dataset
test_dataset = SkinCancerDataset(root_dir=test_path, transform=transform)

# Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(train_dataset.label_map)  # Number of classes in your dataset
)

# Preprocess function
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Define training arguments with a higher learning rate
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,  # Increase the number of epochs
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,  # Increase the learning rate
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_strategy="epoch",  # Log after every epoch
)

# Custom callback to print accuracy and loss
class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            accuracy = metrics.get('eval_accuracy')
            loss = metrics.get('eval_loss')
            if accuracy is not None and loss is not None:
                print(f"Epoch {state.epoch}: Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'eval_accuracy' in logs and 'eval_loss' in logs:
            print(f"Epoch {state.epoch}: Accuracy: {logs['eval_accuracy']:.4f}, Loss: {logs['eval_loss']:.4f}")

# Custom trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        accuracy = accuracy_score(p.label_ids, preds)
        return {"accuracy": accuracy}

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

# Instantiate the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=CustomDataset(small_train_dataset),
    eval_dataset=CustomDataset(test_dataset),  # Evaluate on the test dataset to check for overfitting
    data_collator=data_collator,
    tokenizer=feature_extractor,
    compute_metrics=lambda p: {"accuracy": accuracy_score(np.argmax(p.predictions, axis=1), p.label_ids)},
    callbacks=[PrintMetricsCallback()],  # Add the custom callback
)

# Train the model
trainer.train()

# Evaluate the model on the training data to check for overfitting
results_train = trainer.evaluate(eval_dataset=CustomDataset(small_train_dataset))
print("Training Set Evaluation:", results_train)

# Evaluate the model on the test data to check for generalization
results_test = trainer.evaluate(eval_dataset=CustomDataset(test_dataset))
print("Test Set Evaluation:", results_test)

# Function to reverse normalization
def reverse_normalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    return tensor * std + mean

# Generate detailed evaluation report
def evaluate_model(trainer, dataset, prefix):
    # Get predictions
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Classification report
    report = classification_report(labels, preds, target_names=[key for key in train_dataset.label_map])
    print(report)

    # Save classification report to file
    with open(f"{prefix}_classification_report.txt", "w") as f:
        f.write(report)

    # Save some example predictions
    correct_preds = np.where(preds == labels)[0]
    incorrect_preds = np.where(preds != labels)[0]

    def save_examples(indices, prefix):
        for i in indices[:10]:  # Save first 10 examples
            img_path = train_dataset.images[i]
            img = Image.open(img_path)
            img_tensor = transform(img).unsqueeze(0)
            img_tensor = reverse_normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = img_tensor.squeeze().permute(1, 2, 0).numpy()
            plt.imshow(img_tensor)
            plt.title(f"True: {labels[i]}, Pred: {preds[i]}")
            plt.savefig(f"{prefix}_example_{i}.png")
            plt.close()

    save_examples(correct_preds, f"{prefix}_correct")
    save_examples(incorrect_preds, f"{prefix}_incorrect")

# Evaluate and save reports for training and test sets
evaluate_model(trainer, CustomDataset(small_train_dataset), "train")
evaluate_model(trainer, CustomDataset(test_dataset), "test")
