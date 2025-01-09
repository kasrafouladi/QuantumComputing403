import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# تنظیمات اولیه
z_dim = 100  # ابعاد ورودی نویز تصادفی برای مولد
lr = 0.0002  # نرخ یادگیری
batch_size = 64  # اندازه بچ در هر مرحله آموزش
epochs = 50  # تعداد اپوک‌ها برای آموزش

# آماده‌سازی داده‌های MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

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

# ساخت مدل‌ها
generator = Generator()
discriminator = Discriminator()

# استفاده از BCELoss به عنوان تابع هزینه
criterion = nn.BCELoss()

# استفاده از Adam optimizer برای هر دو مدل
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# بررسی وجود پوشه ذخیره تصاویر
if not os.path.exists('generated_images'):
    os.makedirs('generated_images')

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# تابع برای ذخیره وضعیت مدل‌ها
def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch):
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'epoch': epoch
    }
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch + 1}.pth')
    print(f"Checkpoint saved at epoch {epoch + 1}", flush=True)

# تابع برای بارگذاری وضعیت مدل‌ها
def load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, filename):
    checkpoint = torch.load(filename, weights_only=True)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filename}, resuming from epoch {epoch+1}", flush=True)
    return epoch

# لود کردن وضعیت از چک‌پوینت قبلی
checkpoint_files = sorted(os.listdir('checkpoints'))
if checkpoint_files:
    latest_checkpoint = checkpoint_files[-1]
    epoch = load_checkpoint(generator, discriminator, optimizer_g, optimizer_d, f'checkpoints/{latest_checkpoint}') + 1
else:
    epoch = 0  # شروع از اپوک اول


log_file = open("./log.txt", 'a')

epochs = max(epochs, epoch + 1)

# شروع آموزش از اپوک جاری
ep = epoch
while ep < epochs:
    t = time.time()
    for i, (real_images, _) in enumerate(trainloader):
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)  # برچسب‌های واقعی
        fake_labels = torch.zeros(batch_size, 1)  # برچسب‌های جعلی
        
        # --- آموزش تفکیک‌کننده --- 
        optimizer_d.zero_grad()
        
        # آموزش با داده‌های واقعی
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        
        # تولید داده‌های جعلی
        noise = torch.randn(batch_size, z_dim)  # نویز تصادفی
        fake_images = generator(noise)
        
        # آموزش با داده‌های جعلی
        outputs = discriminator(fake_images.detach())  # عدم به‌روزرسانی گرادیان مولد
        d_loss_fake = criterion(outputs, fake_labels)
        
        # مجموع خطاهای تفکیک‌کننده
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # --- آموزش مولد --- 
        optimizer_g.zero_grad()
        # آموزش مولد
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # مولد می‌خواهد تفکیک‌کننده را فریب دهد
        g_loss.backward()
        optimizer_g.step()

    # چاپ نتایج و ذخیره تصاویر هر چند اپوک
    if (ep + 1) % 10 == 0:
        print(f'Epoch [{ep+1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}', flush=True)
        log_file.write(f'Epoch [{ep+1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}\n')
        log_file.flush()  # flush کردن فایل پس از نوشتن
        # ذخیره تصاویر تولیدی
        with torch.no_grad():
            z = torch.randn(64, z_dim)
            generated_images = generator(z).detach().cpu()
            generated_images = generated_images.view(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(generated_images, nrow=8, padding=2, normalize=True)
            grid_img = grid.permute(1, 2, 0).numpy()
            plt.imshow(grid_img)
            plt.axis('off')
            plt.savefig(f"generated_images/generated_{ep+1}.png")
            plt.close()

    print(f'{int(time.time() - t + 1)} sec', flush=True)

    if ep + 1 == epochs:
        k = int(input(f'Epoch [{ep + 1}/{epochs}] completed.\nDo you want to continue k more epochs? (yes:enter the value of k/no: enter 0): '))
        epochs += k

    ep += 1

# سوال از کاربر برای ذخیره وضعیت مدل
save_choice = input('Do you want to save the model checkpoint? (yes/no): ')
if save_choice.lower() == 'yes':
    save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epochs - 1)  # ذخیره در آخرین اپوک
    print("Checkpoint saved successfully!", flush=True)
else:
    print("Checkpoint not saved.", flush=True)

log_file.close()