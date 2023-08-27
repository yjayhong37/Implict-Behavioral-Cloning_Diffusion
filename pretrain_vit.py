
"""
Updated. Learning Rate Scheduling and Batch Normalization, Gradient Clipping, Early Stopping 
"""
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import DeiTFeatureExtractor, DeiTForImageClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import numpy as np
import timm

# Define constants
BATCH_SIZE = 32
NUM_CLASSES = 1  # Suturing
# OUTPUT_DIR = "/home/yeongjun/DEV/ibc_diffusion/pretraining_output"
OUTPUT_DIR='/Users/alan/DEV/ibc_diffusion/pretraining_output'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f.path for f in os.scandir(root_dir) if f.is_file()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = 0  # Suturing
        return image, label

# Load dataset and create DataLoader
# dataset = CustomDataset(root_dir="/home/yeongjun/DEV/ibc_diffusion/dataset/preprocessed", transform=transform)
dataset = CustomDataset(root_dir='/Users/alan/DEV/ibc_diffusion/dataset/preprocessed',transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load pretrained ViT model from timm
model_name = "vit_base_patch16_224"
model = timm.create_model(model_name, pretrained=True)
num_features = model.head.in_features
model.head = torch.nn.Linear(num_features, NUM_CLASSES)
model.train()

# Pretraining loop
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

criterion = torch.nn.CrossEntropyLoss()

num_epochs = 30
losses = []
action_embeddings = []

# Early stopping parameters
patience = 5
min_loss = float('inf')
early_stop_counter = 0

batch_norm = torch.nn.BatchNorm2d(num_features)  # Adding batch normalization

dropout = torch.nn.Dropout(0.2)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        images = dropout(images)

        # outputs = model(pixel_values=images).logits
        # outputs = model(images).logits
        outputs=model(images)
        loss = criterion(outputs, labels)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        torch.cuda.empty_cache()

        # Extract and accumulate embeddings
        with torch.no_grad():
            features = model.head(model.forward_features(images))
            embeddings = features.cpu().numpy()
            action_embeddings.extend(embeddings)

    # Learning rate scheduler step
    scheduler.step(epoch_loss)

    # Early stopping check
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")


timestamp = int(time.time())


subfolder_name = f"pretraining_{timestamp}"
subfolder_path = os.path.join(OUTPUT_DIR, subfolder_name)
os.makedirs(subfolder_path, exist_ok=True)

# Save action embeddings
embeddings_save_path = os.path.join(subfolder_path, "action_embeddings.npy")
np.save(embeddings_save_path, action_embeddings)

# Save training log
log_filename = os.path.join(subfolder_path, f"training_log_{timestamp}_{epoch}.txt")
with open(log_filename, 'w') as log_file:
    for epoch, loss in enumerate(losses):
        log_file.write(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss:.4f}\n")

# Plot and save loss graph
plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
loss_plot_path = os.path.join(subfolder_path, f"loss_plot_{timestamp}_epochs{epoch}.png")
plt.savefig(loss_plot_path)

# Save the trained ViT model
model_save_path = os.path.join(subfolder_path, f"pretrained_deit_suturing_{timestamp}_epochs{epoch}.pt")
torch.save(model.state_dict(), model_save_path)

print("Training completed. Model, embeddings, logs, and plots saved.")
