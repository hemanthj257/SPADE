import torch.nn as nn

# Define SPADE Block
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        
        self.mlp_shared = nn.Conv2d(label_nc, 128, kernel_size=3, padding=1)
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        actv = nn.ReLU()(self.mlp_shared(segmap))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta

# Define SPADE Generator
class SPADEGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=3, segmap_nc=1):
        super(SPADEGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.spade1 = SPADE(64, segmap_nc)
        self.spade2 = SPADE(64, segmap_nc)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_nc, kernel_size=3, padding=1),
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, x, segmap):
        x = self.encoder(x)
        x = self.spade1(x, segmap)
        x = self.spade2(x, segmap)
        x = self.decoder(x)
        return x

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)