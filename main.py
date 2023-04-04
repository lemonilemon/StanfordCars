import matplotlib.pyplot as plt
import time
import os
import numpy as np
from collections import OrderedDict
import math
import torch
from torch import nn
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

if not os.path.exists(os.path.dirname("./checkpoints")):
    os.makedirs("./checkpoints")
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False):
        super().__init__()
        self.downsample = downsample
        self.normalpath = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size = 3, stride = 2 if downsample else 1, padding = 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size = 1, stride = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 2 if downsample else 1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
       return self.normalpath(x) + self.shortcut(x)

class ResNet50(nn.Module):
  def __init__(self, resblock, in_channels, out_features):
    super().__init__() 
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    self.conv2 = nn.Sequential(
        nn.MaxPool2d(kernel_size = 3, stride = 2), 
    )  
    self.conv2.add_module("conv2_1", resblock(64, 256))
    for i in range(1, 3):
        self.conv2.add_module(f"conv2_{i + 1}", resblock(256, 256))

    self.conv3 = nn.Sequential()
    self.conv3.add_module("conv3_1", resblock(256, 512, downsample = True))
    for i in range(1, 4):
        self.conv3.add_module(f"conv3_{i + 1}", resblock(512, 512))

    self.conv4 = nn.Sequential()
    self.conv4.add_module("conv4_1", resblock(512, 1024, downsample = True))
    for i in range(1, 6):
        self.conv4.add_module(f"conv4_{i + 1}", resblock(1024, 1024))

    self.conv5 = nn.Sequential()
    self.conv5.add_module("conv5_1", resblock(1024, 2048, downsample = True))
    for i in range(1, 3):
        self.conv5.add_module(f"conv5_{i + 1}", resblock(2048, 2048))

    self.gap = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = torch.nn.Linear(2048, out_features)

  @autocast()
  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    out = self.gap(out)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out

    
LEARNING_RATE = 0.004
MOMENTUM = 0.5
EPOCHS = 150 
BATCH_SIZE = 128 
IMAGE_SIZE = 224 
NUM_WORKER = 8 
CHECKPOINT = "./checkpoints/Tue Apr  4 22:59:33 2023"

model = ResNet50(Resblock, 3, 196)
model = model.to(device)
print(model)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(35),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomPosterize(bits=2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
scaler = GradScaler()
criterion = nn.CrossEntropyLoss()

if CHECKPOINT == "": 
    cur_epoch = 0
    datapoints = [[], []]
else: 
    print(f"Loading data from {CHECKPOINT}")
    load = torch.load(CHECKPOINT + "/save.pt")
    optimizer.load_state_dict(load["optimizer_state_dict"])
    model.load_state_dict(load["model_state_dict"])
    scaler.load_state_dict(load["scaler_state_dict"])
    cur_epoch = load["epoch"]
    datapoints = load["datapoints"]

model.train()

train_data = torchvision.datasets.StanfordCars(root = "./data", split = "train", transform = train_transform, download = True)
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
test_data = torchvision.datasets.StanfordCars(root = "./data", split = "test", transform = test_transform, download = True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

datasize = len(train_data)
n_iterations = math.ceil(datasize / BATCH_SIZE)

plt.ion()
plt.figure(1)
plt.style.use('classic')
plt.title("Loss curve")
plt.xlabel("# of Epochs")
plt.ylabel("Loss")
plt.plot(datapoints[0], datapoints[1])
plt.plot(datapoints[0], datapoints[1])
plt.draw()
plt.pause(0.01)

for epoch in range(cur_epoch, EPOCHS):
  avg_loss = 0
  for i, [images, labels] in enumerate(train_dataloader):
    optimizer.zero_grad()
    images = images.to(device)
    labels = labels.to(device)
    with autocast():
      pred = model(images)
      loss = criterion(pred, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    avg_loss += loss.item() * images.shape[0]
    if (i + 1) % 10 == 0 or (i + 1) == n_iterations:
      print(f"Loss: {loss:.7f} [iteration {i + 1}/{n_iterations} in epoch {epoch + 1}/{EPOCHS}]")
  avg_loss /= datasize   

  datapoints[0].append(epoch + 1)
  datapoints[1].append(avg_loss)
  plt.clf()
  plt.style.use('classic')
  plt.title("Loss curve")
  plt.xlabel("# of Epochs")
  plt.ylabel("Loss")
  plt.plot(datapoints[0], datapoints[1])
  plt.draw()
  plt.pause(0.01)
  if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
    checkpoint = {
        'epoch' : epoch + 1,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'datapoints' : datapoints
    }
    filedir = "./checkpoints/" + time.asctime()
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    torch.save(checkpoint, filedir + "/save.pt")
    plt.savefig(filedir + "/loss_curve.png")
    
print("Training Complete!")

model.eval()

correct = 0
with torch.no_grad():
  for i, [images, labels] in enumerate(test_dataloader):
    images = images.to(device)
    labels = labels.to(device)
    with autocast():
      pred = model(images)
      loss = criterion(pred, labels)
    _, guess = torch.max(pred, dim = 1)
    correct += torch.sum(guess == labels)
print(f"Accracy: {correct / len(test_data) * 100}% in {len(test_data)} tests")


