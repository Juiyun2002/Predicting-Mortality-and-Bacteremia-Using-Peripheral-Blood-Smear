import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.optim as optim
from torch.nn.functional import sigmoid
from sklearn.metrics import accuracy_score, f1_score

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
else:
    DEVICE = torch.device("cpu")

class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform, aug_transform=None):
        self.labels_df = df
        self.img_dir = img_dir
        self.transform = transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.labels_df.iloc[idx, 1])


        image = self.transform(image)

        # if self.transform:
        #     image = self.transform(image)

        return image, label
    
aug_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Or (224, 224) for ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



def run(DAYS):
    lab_data_train = pd.read_csv('/home/jupyter-juiyun/Digital_Health/data/lab_data_train.csv')
    lab_data_test = pd.read_csv('/home/jupyter-juiyun/Digital_Health/data/lab_data_test.csv')

    WBC_train = lab_data_train[['file_name', 'event[all-cause mortality]', 'time[all-cause mortality]']].copy().dropna()
    WBC_test = lab_data_test[['file_name', 'event[all-cause mortality]', 'time[all-cause mortality]']].copy().dropna()

    def mortality_data(df_data, days):
        file_names = []
        labels = []

        for i in range(len(df_data)):
            file_name = df_data['file_name'][i]
            event = df_data['event[all-cause mortality]'][i]
            time = df_data['time[all-cause mortality]'][i]

            if time <= days:
                if event == 1:
                    file_names.append(file_name)
                    labels.append(1)

            else:
                file_names.append(file_name)
                labels.append(0)

        result = pd.DataFrame({'file_name': file_names, 'label': labels})
        return result
    

    WBC_train.reset_index(drop=True, inplace=True)
    WBC_test.reset_index(drop=True, inplace=True)
    data_train = mortality_data(WBC_train, DAYS)
    data_val = mortality_data(WBC_test, DAYS)

    dir = '/home/jupyter-juiyun/~backup_link/PBS'
    train_dataset = CustomImageDataset(data_train, dir, transform=transform)
    val_dataset = CustomImageDataset(data_val, dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)


     # ----- Model Setup -----
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1)  # Single continuous output
    model = model.to(DEVICE)  # Move to GPU

    # ----- Loss & Optimizer -----
    criterion = nn.BCEWithLogitsLoss()  # or nn.L1Loss() for MAE
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # ----- Training Loop -----
    num_epochs = 20 

    for epoch in range(num_epochs):
        # ========== Training ==========
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            #outputs = torch.sigmoid(outputs)  # Apply sigmoid if using BCELoss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # ========== Validation ==========
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)

                outputs = model(images)
                #outputs = torch.sigmoid(outputs)  # Apply sigmoid if using BCELoss
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader)

        # Convert outputs to binary (0 or 1)
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_labels_binary = np.array(val_labels).astype(int)

        val_accuracy = accuracy_score(val_labels_binary, val_preds_binary)
        val_f1 = f1_score(val_labels_binary, val_preds_binary)

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss (BCE): {avg_train_loss:.4f}")
        print(f"  Val Loss (BCE):   {avg_val_loss:.4f}")
        print(f"  Val Accuracy:     {val_accuracy:.4f}")
        print(f"  Val F1 Score:     {val_f1:.4f}\n")


    # Save val_preds and val_labels
    val_preds = np.array(val_preds).flatten()
    val_labels = np.array(val_labels).flatten()
    np.save('/home/jupyter-juiyun/Digital_Health/resnet_code/densenet_'+str(DAYS)+'d_preds.npy', val_preds)
    np.save('/home/jupyter-juiyun/Digital_Health/resnet_code/densenet_'+str(DAYS)+'d_labels.npy', val_labels_binary)


if __name__ == "__main__":
    for d in [1,5,7,30, 3]:
        run(d)
        print(f"Finished processing for {d} days.\n")

