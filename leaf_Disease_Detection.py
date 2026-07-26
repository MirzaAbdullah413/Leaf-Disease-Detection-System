import copy
import torch
import zipfile
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from tqdm.auto import tqdm
from google.colab import drive

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Unzip dataset locally
zip_path = "/content/drive/MyDrive/PlantVillage.zip"
extract_path = "/content/PlantVillage"

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
else:
    print("Already extracted, skipping.")

# Check structure before running main() — comment this out once confirmed
!find /content/PlantVillage -maxdepth 3 -type d
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    # Path to your zip file in Drive
    zip_path = "/content/drive/MyDrive/PlantVillage.zip"

    #Loading the DataSet
    train_dataset = datasets.ImageFolder(
        r"/content/PlantVillage/PlantVillage/train",
        transform = transforms.ToTensor()

    )
    val_dataset = datasets.ImageFolder(
        r"/content/PlantVillage/PlantVillage/val",
        transform = transforms.ToTensor()
    )

    dataset = ConcatDataset([train_dataset, val_dataset])
    print(len(dataset))

    #Initializing three tensors of shape (3,) with zeros
    mean = torch.zeros(3)
    std = torch.zeros(3)

    #computing the mean and standard deviation
    for image, _ in tqdm(dataset, desc="Calculating"):
        mean += image.mean((1, 2))
        std += image.std((1, 2))

    mean /= len(dataset)
    std /= len(dataset)

    # mean = torch.tensor([0.4664, 0.4891, 0.4104])
    # std = torch.tensor([0.1761, 0.1500, 0.1925])

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")

    #Transforming the data
    def data_transformations(mean, std):
        #this transformation will happen on the data for training
        train_transformation = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        #Transformation for validation data
        val_transformation = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return train_transformation, val_transformation

    train_transformation, val_transformation = data_transformations(mean, std)
    train_dataset.transform = train_transformation
    val_dataset.transform = val_transformation

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    #making a CNN Block 
    class CNNBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size = 3, padding=1):
            #initilize the parent nn.Module class
            super(CNNBlock, self).__init__()

            self.block = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= kernel_size, padding = padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        def forward(self, x):
            return self.block(x)

    #making a CNN
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
                nn.Linear(512*7*7, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, num_classes)
            )
        def forward(self, x):
            x=self.conv_block1(x)
            x=self.conv_block2(x)
            x=self.conv_block3(x)
            x=self.conv_block4(x)
            x=self.conv_block5(x)
            x=self.classifier(x)
            return x

    num_classes = len(train_dataset.classes)
    print(num_classes)


    #instantiating the model
    model = CNN(num_classes)

    #Loss Function
    loss_function = nn.CrossEntropyLoss()

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

#Method for training
    def train_epoch(model, train_loader, loss_function, optimizer, device):
        model.train()
        running_loss = 0.0
        #Iterate over the batches of data
        for images,classes in tqdm(train_loader, desc="Training", leave=False):
            images,classes = images.to(device), classes.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, classes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss/len(train_loader.dataset)
        return epoch_loss

    #Method for Validating
    def validate_epoch(model, val_loader, loss_function, device):
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, classes in val_loader:
                images, classes = images.to(device), classes.to(device)
                outputs = model(images)
                val_loss = loss_function(outputs, classes)
                running_val_loss += val_loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += images.size(0)
                correct += (predicted==classes).sum().item()
        epoch_val_loss = running_val_loss/len(val_loader.dataset)
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
            print(f"Epoch[{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.4f}")

            if epoch_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_accuracy
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())

        print("-----Finished Training------")
        if best_model_state:
            print(f"\n---Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch}---")
            model.load_state_dict(best_model_state)
        metrics = [train_losses, val_losses, val_accuracies]
        return model, metrics

    #Start the training process by calling the training_loop method
    trained_model, training_metrics = training_loop(
        model = model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=20,
        device= device
    )

if __name__ == '__main__':
    main()
        






