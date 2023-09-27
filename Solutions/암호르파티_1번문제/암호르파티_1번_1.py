# %% [markdown]
# 본 코드는 ImageNet 데이터에 대하여 PGD 기반의 Perturbation 공격을 수행하는 과정과 결과를 보여준다.

# %% [markdown]
# # Dependencies and Imports

# %%
# Install requirements
# %pip install torchattacks

# Create a directory to store our results
# ! mkdir 

# %%
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights,ResNet34_Weights,ResNet50_Weights,ResNet101_Weights,ResNet152_Weights
import torchvision.datasets as dsets

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torchattacks
from torchattacks import PGD

# %% [markdown]
# # Display Utils

# %%
# Functions to show an image
def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True).permute(1,2,0)
    plt.imshow(img)
    plt.title(title)
    plt.show()

# %% [markdown]
# # Load Model and Data

# %% [markdown]
# 공격 대상이 되는 객체 분류용 AI 모델로는 pre-trained ResNet-152를 사용하였다.
# 해당 모델은 ImageNet LSVRC 2012 Validation Set에 대해 약 82.1%의 분류 정확도를 보였다.

# %%
# Load a pretrained model
# model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
# or any of these variants
# model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
# model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
# model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
model.eval()

# %%
# Data augmentation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Load ImageNet Data
DSET_SIZE=50000
TRAIN_SIZE=4000
TEST_SIZE=1000

index = np.arange(DSET_SIZE)
np.random.shuffle(index)
train_index = index[:TRAIN_SIZE]
test_index = index[TRAIN_SIZE: (TRAIN_SIZE + TEST_SIZE)]

BATCH_SIZE=256

train_set = dsets.ImageFolder(root='./ILSVRC2012_img_val', transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_index), shuffle=False)
test_set = dsets.ImageFolder(root='./ILSVRC2012_img_val', transform=preprocess)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(test_index), shuffle=False)

# Download ImageNet labelss
# %wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# move the model to GPU for speed if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# %%
# Test model accuracy
correct,total=0,0
with torch.no_grad():
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1, keepdim = True)
        correct += preds.eq(labels.view_as(preds)).sum()
        total += labels.shape[0]
print(f"model accuracy: {100 * correct / total}")

# %% [markdown]
# # Attack

# %% [markdown]
# 아래의 코드는 ImageNet 데이터에 있는 hard disc 이미지에 대해 Perturbation 공격을 수행하였더니
# 본래의 이미지는 hard disc라고 인식하는 반면, Perturbated 이미지는 스컹크로 인식함을 보여준다.  
# 본래의 이미지와 Perturbated 이미지는 육안으로 거의 구분될 수 있음을 확인 할 수 있다.

# %%
# Choose a baseline image
image_set = dsets.ImageFolder(root='./ILSVRC2012_img_val', transform=preprocess)
image_loader = torch.utils.data.DataLoader(image_set, batch_size=1,shuffle=True)
imageiter = iter(image_loader)
images, labels = next(imageiter)
images=images.to(device)
labels=labels.to(device)

# %%
# Orinal prediction
imshow(images,categories[labels.item()])
with torch.no_grad():
    outputs = model(images)
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# Show top categories of original image
top5_prob, top5_catid = torch.topk(probabilities, 5)
x = np.arange(5)
top5_categories=[]
top5_probabilities=[]
for i in range(top5_prob.size(0)):
    top5_categories.append(categories[top5_catid[i]]) 
    top5_probabilities.append(top5_prob[i].item())
plt.bar(x, top5_probabilities)
plt.xticks(x, top5_categories)
plt.show()



# %% [markdown]
# 

# %%
# Perform an attack
atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
atk.set_normalization_used(mean=MEAN, std=STD)
print('Perturbation Attack')
print(atk)
perturbated_images = atk(images, labels)

# Perturbated prediction
imshow(perturbated_images,'perturbated '+categories[labels.item()])
with torch.no_grad():
    outputs = model(perturbated_images)
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# Show top categories of perturbated image
top5_prob, top5_catid = torch.topk(probabilities, 5)
x = np.arange(5)
top5_categories=[]
top5_probabilities=[]
for i in range(top5_prob.size(0)):
    top5_categories.append(categories[top5_catid[i]]) 
    top5_probabilities.append(top5_prob[i].item())
plt.bar(x, top5_probabilities)
plt.xticks(x, top5_categories)
plt.show()

# %%



