import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from torchvision.models import ResNet50_Weights, resnet50
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard import program
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

# Constants
EPOCHS = 30 
BATCH_SIZE = 32
NUM_WORKER = 8
IMAGE_SIZE = 224
LOG_DIR = "./runs"

# Tensorboard
writer = SummaryWriter(LOG_DIR)
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', LOG_DIR])
url = tb.launch()
print(f"Tensorflow listening on {url}")

# Transforms
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

# Data
train_data = datasets.StanfordCars(root = "./data", split = "train", transform = train_transform, download = True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
test_data = datasets.StanfordCars(root = "./data", split = "test", transform = test_transform, download = True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)

sample_imgs, sample_labels = next(iter(train_dataloader))
img_grid = torchvision.utils.make_grid(sample_imgs)
writer.add_image("StanfordCarsImages", img_grid)

# Model & optimizer ...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights = ResNet50_Weights.DEFAULT)
model.to(device)
optimizer = torch.optim.Adadelta(model.parameters())
scaler = GradScaler()
criterion = nn.CrossEntropyLoss()
# Model Graph
writer.add_graph(model = model, input_to_model = sample_imgs.to(device))

print("Training Start!")

# Initialize
train_loss = 0.0
train_top1accuracy = 0.0
train_top5accuracy = 0.0
test_loss = 0.0
test_top1accuracy = 0.0
test_top5accuracy = 0.0

# Iterate through epochs
for epoch in range(1, EPOCHS + 1):
    print(f"In epoch {epoch}/{EPOCHS}:")
    # Monitor loss
    train_loss = 0.0
    train_top1accuracy = 0.0
    train_top5accuracy = 0.0
    test_loss = 0.0
    test_top1accuracy = 0.0
    test_top5accuracy = 0.0

    # Train the model
    model.train()
    loop = tqdm(train_dataloader)
    loop.set_description(f"Training Epoch[{epoch}/{EPOCHS}]")
    for images, labels in loop:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        with autocast():
            pred = model(images)
            loss = criterion(pred, labels)
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Calculate Loss & Accuracy
        train_loss += loss.item() * images.shape[0]
        _, indices = torch.max(pred, 1)
        train_top1accuracy += torch.sum(indices == labels)
        _, indices = torch.topk(pred, 5, 1)
        train_top5accuracy += torch.sum(indices == torch.unsqueeze(labels, 1), (0, 1))


    # Evaluate the model
    loop = tqdm(test_dataloader)
    loop.set_description(f"Testing Epoch[{epoch}/{EPOCHS}]")
    model.eval() 
    with torch.no_grad():
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            with autocast():
                pred = model(images)
                loss = criterion(pred, labels)
            # Calculate Loss & Accuracy
            test_loss += loss.item() * images.shape[0]
            _, indices = torch.max(pred, 1)
            test_top1accuracy += torch.sum(indices == labels)
            _, indices = torch.topk(pred, 5, 1)
            test_top5accuracy += torch.sum(indices == torch.unsqueeze(labels, 1), (0, 1))
    
    # Calculate & Save statistic
    for name, param in model.named_parameters():
        writer.add_histogram(tag = name + "_grad", values = param.grad, global_step = epoch)
        writer.add_histogram(tag = name + "_data", values = param.data, global_step = epoch)

    train_loss /= len(train_data)
    train_top1accuracy /= len(train_data)
    train_top5accuracy /= len(train_data)
    test_loss /= len(test_data)
    test_top1accuracy /= len(test_data)
    test_top5accuracy /= len(test_data)
    print(f"In Training set: Loss = {train_loss:.8f}, Top1-accuracy = {train_top1accuracy * 100:.2f}%, Top5-accuracy = {train_top5accuracy * 100:.2f}%")
    print(f"In Test set: Loss = {test_loss:.8f}, Top1-accuracy = {test_top1accuracy * 100:.2f}%, Top5-accuracy = {test_top5accuracy*100:.2f}%")
    writer.add_scalars("Loss", tag_scalar_dict = {"Train" : train_loss, "Test" : test_loss}, global_step = epoch)
    writer.add_scalars("Top1-Accuracy", tag_scalar_dict = {"Train" : train_top1accuracy,
                                                        "Test" : test_top1accuracy}, global_step = epoch)
    writer.add_scalars("Top5-Accuracy", tag_scalar_dict = {"Train" : train_top5accuracy,
                                                        "Test" : test_top5accuracy}, global_step = epoch)

# Summary
writer.add_hparams({
        "epoch" : EPOCHS, 
        "batchsize" : BATCH_SIZE,
        "model" : "ResNet50"
    }, {
        "Loss" : test_loss,
        "Top1-Accuracy" : test_top1accuracy,
        "Top5-Accuracy" : test_top5accuracy
    }
)

writer.close()
