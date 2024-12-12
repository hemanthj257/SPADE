import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import SAROpticalDataset
from model import SPADEGenerator, Discriminator
from perceptual_loss import PerceptualLoss

# Parameters
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Load Data
dataset = SAROpticalDataset("images/sar_train/", "images/oi_train/", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Number of training samples: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")

# Initialize Model, Loss, Optimizer
model = SPADEGenerator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
criterion = nn.MSELoss()
perceptual_loss = PerceptualLoss().to(DEVICE)
optimizer_G = optim.Adam(model.parameters(), lr=LR)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)

# Set starting epoch
START_EPOCH = 0  # Change this to resume from a specific epoch

# Load checkpoint if resuming training
if START_EPOCH > 0:
    model.load_state_dict(torch.load(f"model_epoch_{START_EPOCH}.pth"))
    optimizer_G.load_state_dict(torch.load(f"optimizer_G_epoch_{START_EPOCH}.pth"))
    discriminator.load_state_dict(torch.load(f"discriminator_epoch_{START_EPOCH}.pth"))
    optimizer_D.load_state_dict(torch.load(f"optimizer_D_epoch_{START_EPOCH}.pth"))

# Training Loop
for epoch in range(START_EPOCH, EPOCHS):
    model.train()
    discriminator.train()
    total_loss_G = 0
    total_loss_D = 0

    for sar, optical in dataloader:
        sar = sar.float().to(DEVICE)
        optical = optical.float().to(DEVICE)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_output = discriminator(optical)
        fake_output = discriminator(model(sar, sar).detach())
        loss_D_real = criterion(real_output, torch.ones_like(real_output))
        loss_D_fake = criterion(fake_output, torch.zeros_like(fake_output))
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_output = model(sar, sar)
        loss_G_MSE = criterion(fake_output, optical)
        loss_G_perceptual = perceptual_loss(fake_output, optical)
        loss_G_adv = criterion(discriminator(fake_output), torch.ones_like(real_output))
        loss_G = loss_G_MSE + 0.1 * loss_G_perceptual + 0.001 * loss_G_adv
        loss_G.backward()
        optimizer_G.step()

        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss G: {total_loss_G/len(dataloader):.4f}, Loss D: {total_loss_D/len(dataloader):.4f}")

    # Save the model and optimizer state every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        torch.save(optimizer_G.state_dict(), f"optimizer_G_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")
        torch.save(optimizer_D.state_dict(), f"optimizer_D_epoch_{epoch+1}.pth")

# Save the final trained model
torch.save(model.state_dict(), "sar_colorization_spade.pth")