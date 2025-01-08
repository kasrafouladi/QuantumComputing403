import torch
import torchvision
import torchvision.transforms as transforms
import os

# آماده‌سازی داده‌های MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)

# جایگزینی تمام تصاویر با تصاویر حاوی صفر
for i in range(len(trainset.data)):
    trainset.data[i] = torch.zeros_like(trainset.data[i])

# ذخیره دیتاست جدید
torch.save(trainset, 'modified_mnist.pth')

print("All images in the dataset have been replaced with images of zeros and saved as 'modified_mnist.pth'.")
