import ssl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import ResNetForImageClassification
import time

ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

trainset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

device = torch.device('cpu')

model = ResNetForImageClassification.from_pretrained(
    "microsoft/resnet-18",
    ignore_mismatched_sizes=True,
    num_labels=10,
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 3

print(f"Device: {device}")
print(f"Training on MNIST for {num_epochs} epochs\n")

for epoch in range(num_epochs):
    epoch_start = time.time()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_loss /= test_total
    test_acc = test_correct / test_total
    epoch_time = time.time() - epoch_start

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f}  "
          f"test_loss: {test_loss:.4f}  test_acc: {test_acc:.4f}  "
          f"time: {epoch_time:.1f}s")

print("\nTraining completed!")
