import torch
import torch.nn as nn

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization for stable adversarial training.
    """
    def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm),
                use_spectral_norm
            ),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm),
                use_spectral_norm
            ),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm),
                use_spectral_norm
            ),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm),
                use_spectral_norm
            ),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm),
                use_spectral_norm
            )
        )

        self.init_weights()

    def init_weights(self, gain=0.02):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]
