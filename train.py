# %%
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 1. Low-level processing and augmentation for Training Set
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Uniform image sizing
    transforms.RandomHorizontalFlip(p=0.5), # Augmentation
    transforms.RandomRotation(degrees=15), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Low-level lighting tweak
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Setup for Validation
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Connect to your Step 2 folder structure
# Make sure the 'wheat_project/dataset/...' folder is inside your current workspace!
train_dataset = ImageFolder(root='C:/Users/prasi/OneDrive/Documents/wheat_project/dataset/train', transform=train_transform)
val_dataset = ImageFolder(root='C:/Users/prasi/OneDrive/Documents/wheat_project/dataset/val', transform=val_test_transform)

# 4. Generate the loaders that the training loop is looking for
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Dataset successfully connected! Found {len(train_dataset)} training images.")
import torch
import torchvision.models as models
import torch.nn as nn

# [Step 1 Code]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 4)
model.classifier = model.classifier.to(device)
print(model.classifier)

# %%
# 1. Unfreeze the whole model first
for param in model.parameters():
    param.requires_grad = True

# 2. (Optional alternative) If training feels too slow on your CPU, 
# you can freeze just the early layers and only unfreeze the deep features:
# for param in model.features[:9].parameters():
#     param.requires_grad = False


# %%
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Gradually decays learning rate over 12 epochs
scheduler = CosineAnnealingLR(optimizer, T_max=12)

# %%
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=12):
    print("Starting training loop with learning rate scheduling...")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = (correct_train / total_train) * 100
        
        # 🔥 CRUCIAL: Step the scheduler here to decay the learning rate!
        scheduler.step()
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100
        
        # Fetch current learning rate to monitor the decay
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} (LR: {current_lr:.6f}) -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

# Trigger the updated training engine with the scheduler passed as an argument
train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=12)

# %%
torch.save(model.state_dict(), "wheat_disease_mobilenet.pth")
print("Weights safely locked into 'wheat_disease_mobilenet.pth'!")

# %%
