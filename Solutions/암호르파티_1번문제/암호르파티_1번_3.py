# %% [markdown]
# 본 코드는  GTSRB 데이터에 대하여 Adversarial Patch를 생성하는 과정과 결과를 보여준다

# %% [markdown]
# # Dependencies and Imports

# %%
# Install requirements
# %pip install

# Create a directory to store our results
# %mkdir 

# %%
import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms
from torchvision import models
import torchvision.datasets as dsets

import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image

# %% [markdown]
# # Display Utils

# %%
# Functions to show an image
def imshow(img, title):
    plt.imshow(np.clip(np.transpose(img, (1, 2, 0)), 0, 1))
    plt.title(title)
    plt.show()

# %% [markdown]
# # Load Model and Data

# %%
# Load a finetuned model
PATH_TO_MODEL = "./Traffic_Sign_Classification.pth"

model = models.resnet18()
in_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(in_ftrs, 43)
model.load_state_dict(torch.load(PATH_TO_MODEL))
model.eval()

# %%
# Data transformation
import torchvision.transforms as transforms
preprocess = transforms.Compose(
    [transforms.Resize([112, 112]),
    transforms.ToTensor()]
)

# Load GTSRB Data
BATCH_SIZE = 1

train_set = torchvision.datasets.ImageFolder(root="./GTSRB/Train", transform=preprocess)

# Divide data into training and validation (0.8 and 0.2)
ratio = 0.8
n_train_examples = int(len(train_set) * ratio)
n_val_examples = len(train_set) - n_train_examples
gtsrb_train_data, val_data = data.random_split(train_set, [n_train_examples, n_val_examples])
train_loader = data.DataLoader(gtsrb_train_data, shuffle=True, batch_size = BATCH_SIZE)
val_loader = data.DataLoader(val_data, shuffle=True, batch_size = BATCH_SIZE)

# %% [markdown]
# # Generate Patch & Attack

# %% [markdown]
# Adversarial Patch의 타겟, 즉 해당 패치를 붙였을 때 분류자 모델이 인식해야할 결과는 Go Straight Traffic Sign으로, Adversarial Patch의 넓이는 이미지의 4%로 설정하였다

# %%
LR = 1.0
EPOCHS = 5
THRESHOLD=0.9
MAX_ITER=100
TARGET=35 # Go Straight Traffic Sign

# Initialize the patch
IMG_SIZE=(3, 112, 112)
NOISE_RATIO=0.2
MASK_SIZE = int((NOISE_RATIO * IMG_SIZE[1] * IMG_SIZE[2])**0.5)

patch = np.random.rand(IMG_SIZE[0], MASK_SIZE, MASK_SIZE)

# %% [markdown]
# Adversarial Patch를 얻기 위해 먼저 패치를 초기화한 후 다음의 함수를 최대화하도록 학습시켰다.
# $$\mathbb{E}x∼X,l∼L [log(Pr(TARGET|A(patch,x,l)))]$$
# A(p, x, l)는 이미지 x의 위치 x에 패치 p를 붙이는 연산자이다.
# 
# ![](./A.png)
# 
# Pr(y|x)는 입력 이미지 x에 대해 분류자 모델이 y로 인식할 확률이다.
# 
# X는 이미지의 훈련 셋이며, L은 이미지의 위치에 대한 분포이다.
# 
# 이를 풀어 설명하자면, 이미지를 뽑으면서 위치를 바꿔가면서 패치를 적용해보아도 이 패치가 여전히 그 모든 전체 결과에 대해서 Adversarial Patch로써 잘 동작할 수 있도록 학습을 시키는 것이다.
# 
# 해당 패치는 99.1%의 공격 성공률을 보였다.

# %%
# Path to save Patch
PATH_TO_PATCH = "./Patch.png"
best_acc = 0.0
best_patch=patch

print("Patch saved at "+PATH_TO_PATCH)
print('')

for epoch in range(EPOCHS):
    print(f"Epoch-{epoch}: ",end='')
    start_time = time.monotonic()

    # Train the patch
    for images,labels in train_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        if preds[0] != labels[0] or preds[0].data.numpy() == TARGET: continue # only train on images that were originally classified successly
        
        for _ in range(4):
            # Apply the patch
            extended_patch = np.zeros(IMG_SIZE)
            x_loc, y_loc = np.random.randint(low=0, high=IMG_SIZE[1]-patch.shape[1]), np.random.randint(low=0, high=IMG_SIZE[2]-patch.shape[2]) # random patch location
            for i in range(patch.shape[0]):
                extended_patch[:, x_loc:x_loc + patch.shape[1], y_loc:y_loc + patch.shape[2]] = patch
            mask = extended_patch.copy()
            mask[mask != 0] = 1.0
            extended_patch = torch.from_numpy(extended_patch)
            mask = torch.from_numpy(mask)
            patched_images = torch.mul(mask.type(torch.FloatTensor), extended_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), images.type(torch.FloatTensor))


            # Optimize the patch
            patched_images = Variable(patched_images.data, requires_grad=True)
            output = model(patched_images)
            loss= -torch.nn.functional.log_softmax(output, dim=1)[0][TARGET]
            loss.backward()
            patch_grad = patched_images.grad.clone()
            patched_images.grad.data.zero_()
            extended_patch = extended_patch.type(torch.FloatTensor) - LR * patch_grad
            extended_patch = torch.clamp(extended_patch, min=-3, max=3)
            extended_patch = extended_patch.numpy()
            patch = extended_patch[0][:, x_loc:x_loc + patch.shape[1], y_loc:y_loc + patch.shape[2]]

        
    # Evaluate the patch
    success,total=0,0
    with torch.no_grad():
        for images,labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            # only evaluate on images that were originally classified successly
            if preds[0] != labels[0]: continue
            total+=1
             # Apply the patch
            extended_patch = np.zeros(IMG_SIZE)
            x_loc, y_loc = np.random.randint(low=0, high=IMG_SIZE[1]-patch.shape[1]), np.random.randint(low=0, high=IMG_SIZE[2]-patch.shape[2]) # random patch location
            for i in range(patch.shape[0]):
                extended_patch[:, x_loc:x_loc + patch.shape[1], y_loc:y_loc + patch.shape[2]] = patch
            mask = extended_patch.copy()
            mask[mask != 0] = 1.0
            extended_patch = torch.from_numpy(extended_patch)
            mask = torch.from_numpy(mask)
            patched_images = torch.mul(mask.type(torch.FloatTensor), extended_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), images.type(torch.FloatTensor))

            patched_outputs = model(patched_images)
            _, patched_preds = torch.max(patched_outputs, dim=1)
            if preds[0].data.numpy() != TARGET and patched_preds[0].data.numpy() == TARGET: success += 1
    end_time = time.monotonic()
    acc = 100 * success / total
    print("Success Rate = %.4f , Time = %.2f seconds" % (acc, end_time - start_time))
    if acc > best_acc:
        best_acc = acc
        best_patch=patch
        plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)), 0, 1))
        plt.savefig(PATH_TO_PATCH)


# %%
imshow(best_patch,"Adversarial Patch")
print("Success Rate = %.4f" % (best_acc))

# %% [markdown]
# 아래의 코드는 GTSRB 데이터에 있는 STOP 이미지에 위에서 만든 Adversarial Patch를 붙였더니 본래의 이미지는 X Sign라고 인식하는 반면, 패치를 붙인 이미지는 Go Straight Sign으로 인식함을 보여준다.
# 

# %%
# Choose a baseline image
image_set = dsets.ImageFolder(root="./GTSRB/Meta", transform=preprocess)
image_loader = torch.utils.data.DataLoader(image_set, batch_size=1,shuffle=False)
imageiter = iter(image_loader)
images, labels = next(imageiter)

# %%
# Orinal prediction
imshow(images.squeeze(0),str(labels.item()))
with torch.no_grad():
    outputs = model(images)
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# Show top categories of original image
top5_prob, top5_catid = torch.topk(probabilities, 5)
x = np.arange(5)
top5_categories=[]
top5_probabilities=[]
for i in range(top5_prob.size(0)):
    top5_categories.append(str(top5_catid[i].item())) 
    top5_probabilities.append(top5_prob[i].item())
plt.bar(x, top5_probabilities)
plt.xticks(x, top5_categories)
plt.show()



# %%
# Apply the patch
extended_patch = np.zeros(IMG_SIZE)
x_loc, y_loc = np.random.randint(low=0, high=IMG_SIZE[1]-best_patch.shape[1]), np.random.randint(low=0, high=IMG_SIZE[2]-patch.shape[2]) # random patch location
for i in range(best_patch.shape[0]):
    extended_patch[:, x_loc:x_loc + best_patch.shape[1], y_loc:y_loc + best_patch.shape[2]] = best_patch
mask = extended_patch.copy()
mask[mask != 0] = 1.0
extended_patch = torch.from_numpy(extended_patch)
mask = torch.from_numpy(mask)
patched_images = torch.mul(mask.type(torch.FloatTensor), extended_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), images.type(torch.FloatTensor))

# Patched prediction
imshow(patched_images.squeeze(0),'patched '+str(labels.item()))
with torch.no_grad():
    outputs = model(patched_images)
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# Show top categories of perturbated image
top5_prob, top5_catid = torch.topk(probabilities, 5)
x = np.arange(5)
top5_categories=[]
top5_probabilities=[]
for i in range(top5_prob.size(0)):
    top5_categories.append(str(top5_catid[i].item())) 
    top5_probabilities.append(top5_prob[i].item())
plt.bar(x, top5_probabilities)
plt.xticks(x, top5_categories)
plt.show()


