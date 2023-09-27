# %% [markdown]
# 본 코드는 전이학습을 이용하여 GTSRB(독일교통표지판) 데이터 분류를 위한 모델을 만드는 과정과 결과를 보여준다.

# %% [markdown]
# # Dependencies and Imports

# %%
# Install requirements
# %pip install

# Create a directory to store our results
# ! mkdir 

# %%
import os
import time

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights
import torchvision.datasets as dsets

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# %% [markdown]
# # Load GTRSB Data

# %%
# Data transformation
import torchvision.transforms as transforms
preprocess = transforms.Compose(
    [transforms.Resize([112, 112]),
    transforms.ToTensor()]
)

# Load Data
BATCH_SIZE = 256

train_set = torchvision.datasets.ImageFolder(root="./GTSRB/Train", transform=preprocess)

# Divide data into training and validation (0.8 and 0.2)
ratio = 0.8
n_train_examples = int(len(train_set) * ratio)
n_val_examples = len(train_set) - n_train_examples
gtsrb_train_data, val_data = data.random_split(train_set, [n_train_examples, n_val_examples])
train_loader = data.DataLoader(gtsrb_train_data, shuffle=True, batch_size = BATCH_SIZE)
val_loader = data.DataLoader(val_data, shuffle=True, batch_size = BATCH_SIZE)

# %% [markdown]
# # Finetune Model

# %% [markdown]
# 모델을 pre-trained ResNet-18으로 초기화한 후 finetuning을 진행하였다.
# 해당 모델은 약 99.9%의 분류 정확도를 보였다.

# %%
# Load a pretrained ResNet model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# or any of these variants
# model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
# model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
# model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
# model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
in_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(in_ftrs, 43) # GTSRB dataset consists of 39209 training images corresponding to 43 classes.

# move the model to GPU for speed if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# %%
# Define a Loss function and optimizer
LR = 0.001
EPOCHS = 15
criterion = torch.nn.CrossEntropyLoss() 
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Path to save model
PATH_TO_MODEL = "./Traffic_Sign_Classification.pth"
best_acc = 0.0

print("Model saved at "+PATH_TO_MODEL)
print('')

# Perform training
train_loss_list = [0]*EPOCHS
train_acc_list = [0]*EPOCHS
val_loss_list = [0]*EPOCHS
val_acc_list = [0]*EPOCHS
for epoch in range(EPOCHS):
    print(f"Epoch-{epoch}:")
    
    # Train the model
    train_start_time = time.monotonic()
    model.train()
    epoch_loss = 0
    epoch_correct,epoch_total=0,0
    for (images, labels) in train_loader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Calculate accuracy
        top_preds = outputs.argmax(1, keepdim = True)
        correct = top_preds.eq(labels.view_as(top_preds)).sum()

        # Optimizing weights
        optimizer.step()

        epoch_loss += loss.item()
        epoch_correct += correct.item()
        epoch_total += labels.shape[0]
    train_end_time = time.monotonic()
    train_loss = epoch_loss
    train_acc = epoch_correct/epoch_total
    
    # Evaluate the model
    val_start_time = time.monotonic()
    model.eval()
    epoch_loss = 0
    epoch_correct,epoch_total=0,0
    with torch.no_grad():
        for (images, labels) in val_loader:
            images=images.to(device)
            labels=labels.to(device)

            # Run predictions
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            top_preds = outputs.argmax(1, keepdim = True)
            correct = top_preds.eq(labels.view_as(top_preds)).sum()

            epoch_loss += loss.item()
            epoch_correct += correct.item()
            epoch_total += labels.shape[0]
    val_end_time = time.monotonic()
    val_loss = epoch_loss
    val_acc = epoch_correct/epoch_total
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), PATH_TO_MODEL)

    train_loss_list[epoch] = train_loss
    train_acc_list[epoch] = train_acc
    val_loss_list[epoch] = val_loss
    val_acc_list[epoch] = val_acc
    
    print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (train_loss, train_acc, train_end_time - train_start_time))
    print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (val_loss, val_acc, val_end_time - val_start_time))


# %%
# val model accuracy
model.load_state_dict(torch.load(PATH_TO_MODEL))
correct,total=0,0
with torch.no_grad():
    for images,labels in val_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1, keepdim = True)
        correct += preds.eq(labels.view_as(preds)).sum()
        total += labels.shape[0]
print(f"model accuracy: {100 * correct / total}")

# %% [markdown]
# 


