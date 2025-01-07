import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

print("C1", flush=True)

# تنظیمات اولیه
z_dim = 100  # ابعاد ورودی نویز تصادفی برای مولد
lr = 0.0002  # نرخ یادگیری
batch_size = 64  # اندازه بچ در هر مرحله آموزش
epochs = 50  # تعداد اپوک‌ها برای آموزش

# آماده‌سازی داده‌های MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

print("C2", flush=True)

# تعریف مولد (Generator)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),  # ورودی 100 عدد تصادفی به لایه 256 نرون
            nn.LeakyReLU(0.2),  # فعال‌سازی با LeakyReLU
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),  # خروجی نهایی 28x28 (تصویر MNIST)
            nn.Tanh()  # تغییر دامنه به [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)  # تغییر به ابعاد تصویر 28x28

print("C3", flush=True)

# تعریف تفکیک‌کننده (Discriminator)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # تخت کردن ورودی به یک بردار
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),  # خروجی یک احتمال (0 یا 1)
            nn.Sigmoid()  # سیگموید برای خروجی احتمال
        )
    
    def forward(self, x):
        return self.model(x)

print("C4", flush=True)

# ساخت مدل‌ها
generator = Generator()
discriminator = Discriminator()

print("C5", flush=True)

# استفاده از BCELoss به عنوان تابع هزینه
criterion = nn.BCELoss()

print("C6", flush=True)

# استفاده از Adam optimizer برای هر دو مدل
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

print("C7", flush=True)

# بررسی وجود پوشه ذخیره تصاویر
if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

print("C8", flush=True)

print(f"len(trainloader) = {len(trainloader)}", flush=True)

# حلقه آموزش GAN
import time

for epoch in range(epochs):
    t = time.time()
    print("-0", flush=True)
    for i, (real_images, _) in enumerate(trainloader):
        #print("1", flush=True)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)  # برچسب‌های واقعی
        fake_labels = torch.zeros(batch_size, 1)  # برچسب‌های جعلی
        #print("2", flush=True)
        # --- آموزش تفکیک‌کننده ---
        optimizer_d.zero_grad()
        #print("3", flush=True)
        # آموزش با داده‌های واقعی
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        #print("4", flush=True)
        # تولید داده‌های جعلی
        noise = torch.randn(batch_size, z_dim)  # نویز تصادفی
        fake_images = generator(noise)
        #print("5", flush=True)
        # آموزش با داده‌های جعلی
        outputs = discriminator(fake_images.detach())  # عدم به‌روزرسانی گرادیان مولد
        d_loss_fake = criterion(outputs, fake_labels)
        #print("6", flush=True)
        # مجموع خطاهای تفکیک‌کننده
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        #print("7", flush=True)
        # --- آموزش مولد ---
        optimizer_g.zero_grad()
        #print("8", flush=True)
        # آموزش مولد
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # مولد می‌خواهد تفکیک‌کننده را فریب دهد
        g_loss.backward()
        optimizer_g.step()
        #print("9", flush=True)
    # چاپ نتایج و ذخیره تصاویر هر 10 اپوک
    print("-1", flush=True)
    if (epoch + 1) % 2 == 0:
        print("1", flush=True)
        print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
        # ذخیره تصاویر تولیدی
        print("2", flush=True)
        """
        with torch.no_grad():
            print("3", flush=True)
            z = torch.randn(64, z_dim)  # تولید نویز برای مولد
            generated_images = generator(z).detach().cpu()
            generated_images = generated_images.view(-1, 28, 28)
            grid = torchvision.utils.make_grid(generated_images, nrow=8, padding=2, normalize=True)  # گرید تصاویر
            print("4", flush=True)
            plt.imshow(grid.permute(1, 2, 0))  # نمایش گرید
            plt.axis('off')  # عدم نمایش محور
            plt.savefig(f"generated_images/generated_{epoch+1}.png")  # ذخیره تصویر
            plt.close()
            print("5", flush=True)
        """
        with torch.no_grad():
            print("3", flush=True)
            z = torch.randn(64, z_dim)  # تولید نویز برای مولد
            generated_images = generator(z).detach().cpu()  # قطع اتصال گرادیان و انتقال به CPU
            generated_images = generated_images.view(-1, 1, 28, 28)  # تغییر شکل به (batch_size, 1, 28, 28)
            grid = torchvision.utils.make_grid(generated_images, nrow=8, padding=2, normalize=True)  # گرید تصاویر

            print("4", flush=True)
            # تبدیل گرید به ابعاد (H, W, C) برای نمایش
            grid_img = grid.permute(1, 2, 0).numpy()  # تبدیل به numpy برای نمایش با plt.imshow
            plt.imshow(grid_img)  # نمایش گرید
            plt.axis('off')  # عدم نمایش محور
            plt.savefig(f"generated_images/generated_{epoch+1}.png")  # ذخیره تصویر
            plt.close()  # بستن پنجره نمایش
    
    print("5", flush=True)

    print("-2", flush=True)
    print(time.time() - t)

print("C9", flush=True)