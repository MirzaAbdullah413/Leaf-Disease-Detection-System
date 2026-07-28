import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# ============================================================
# STEP 0 — RUN THIS FIRST IN A SEPARATE CELL TO CHECK THE PATH
# ============================================================
# Kaggle mounts datasets read-only under /kaggle/input/<dataset-slug>/...
# The exact folder name inside can vary, so confirm it before running main().
#
#   import os
#   for root, dirs, files in os.walk("/kaggle/input"):
#       print(root)
#
# You confirmed premade train/val folders exist inside the PlantVillage
# directory, so point these at them directly.

TRAIN_DIR = "/kaggle/input/plantvillage/PlantVillage/train"  # <-- update after checking structure
VAL_DIR = "/kaggle/input/plantvillage/PlantVillage/val"      # <-- update after checking structure
OUTPUT_DIR = "/kaggle/working"                                # Kaggle's writable output folder
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "leaf_disease_model.pth")


# making a CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


# making a CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_block1 = CNNBlock(3, 32)
        self.conv_block2 = CNNBlock(32, 64)
        self.conv_block3 = CNNBlock(64, 128)
        self.conv_block4 = CNNBlock(128, 256)
        self.conv_block5 = CNNBlock(256, 512)

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.classifier(x)
        return x


def data_transformations(mean, std):
    train_transformation = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transformation, val_transformation


def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)


def validate_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = loss_function(outputs, labels)
            running_val_loss += val_loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += images.size(0)
            correct += (predicted == labels).sum().item()
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total
    return epoch_val_loss, epoch_accuracy


def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    model.to(device)
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    train_losses, val_losses, val_accuracies = [], [], []
    print("------Training Started------")
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)

        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        print(f"Epoch[{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.4f}")

        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

    print("-----Finished Training------")
    if best_model_state:
        print(f"\n---Returning best model with {best_val_accuracy:.2f}% "
              f"validation accuracy, achieved at epoch {best_epoch}---")
        model.load_state_dict(best_model_state)
    metrics = [train_losses, val_losses, val_accuracies]
    return model, metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    # --- Load the premade train/val folders directly ---
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transforms.ToTensor())
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transforms.ToTensor())

    dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Total: {len(dataset)}")

    # --- Compute mean/std across the full dataset (matches original script) ---
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for image, _ in tqdm(dataset, desc="Calculating mean/std"):
        mean += image.mean((1, 2))
        std += image.std((1, 2))
    mean /= len(dataset)
    std /= len(dataset)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")

    train_transformation, val_transformation = data_transformations(mean, std)
    train_dataset.transform = train_transformation
    val_dataset.transform = val_transformation

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(num_classes)

    model = CNN(num_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model, training_metrics = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=20,
        device=device
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({
        "model": trained_model,   # full model object (architecture + weights), not just state_dict
        "classes": class_names,
        "mean": mean,
        "std": std,
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("Go to the notebook's 'Output' / 'Data' tab on the right side of Kaggle "
          "to download leaf_disease_model.pth once the run finishes.")


if __name__ == '__main__':
    main()