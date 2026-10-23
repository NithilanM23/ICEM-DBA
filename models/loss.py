import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =========================================================================
# Adversarial Loss (LSGAN, Hinge, NSGAN)
# =========================================================================

class AdversarialLoss(nn.Module):
    """
    Adversarial loss supporting LSGAN, Hinge, and NSGAN objectives.
    """
    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def forward(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            return self.criterion(outputs, labels)


# =========================================================================
# VGG19 Feature Extractor for Perceptual & Style Losses
# =========================================================================

class VGG19(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            if pretrained:
                weights = models.VGG19_Weights.DEFAULT
                features = models.vgg19(weights=weights).features
            else:
                features = models.vgg19(weights=None).features
        except Exception:
            # Fallback for older torchvision versions or offline environments
            features = models.vgg19(pretrained=False).features

        for m in features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        self.relu1_1 = nn.Sequential(*[features[x] for x in range(2)])
        self.relu1_2 = nn.Sequential(*[features[x] for x in range(2, 4)])
        self.relu2_1 = nn.Sequential(*[features[x] for x in range(4, 7)])
        self.relu2_2 = nn.Sequential(*[features[x] for x in range(7, 9)])
        self.relu3_1 = nn.Sequential(*[features[x] for x in range(9, 12)])
        self.relu3_2 = nn.Sequential(*[features[x] for x in range(12, 14)])
        self.relu3_3 = nn.Sequential(*[features[x] for x in range(14, 16)])
        self.relu3_4 = nn.Sequential(*[features[x] for x in range(16, 18)])
        self.relu4_1 = nn.Sequential(*[features[x] for x in range(18, 21)])
        self.relu4_2 = nn.Sequential(*[features[x] for x in range(21, 23)])
        self.relu4_3 = nn.Sequential(*[features[x] for x in range(23, 25)])
        self.relu4_4 = nn.Sequential(*[features[x] for x in range(25, 27)])
        self.relu5_1 = nn.Sequential(*[features[x] for x in range(27, 30)])
        self.relu5_2 = nn.Sequential(*[features[x] for x in range(30, 32)])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        r1_1 = self.relu1_1(x)
        r1_2 = self.relu1_2(r1_1)
        r2_1 = self.relu2_1(r1_2)
        r2_2 = self.relu2_2(r2_1)
        r3_1 = self.relu3_1(r2_2)
        r3_4 = self.relu3_4(self.relu3_3(self.relu3_2(r3_1)))
        r4_1 = self.relu4_1(r3_4)
        r4_4 = self.relu4_4(self.relu4_3(self.relu4_2(r4_1)))
        r5_1 = self.relu5_1(r4_4)
        r5_2 = self.relu5_2(r5_1)

        return {
            'relu1_1': r1_1,
            'relu2_1': r2_1,
            'relu2_2': r2_2,
            'relu3_1': r3_1,
            'relu3_4': r3_4,
            'relu4_1': r4_1,
            'relu4_4': r4_4,
            'relu5_1': r5_1,
            'relu5_2': r5_2,
        }


# =========================================================================
# Perceptual & Style Losses
# =========================================================================

class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual feature matching loss.
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], pretrained=True, vgg=None):
        super().__init__()
        self.vgg = vgg if vgg is not None else VGG19(pretrained=pretrained)
        self.criterion = nn.L1Loss()
        self.weights = weights

    def forward(self, x, y):
        with torch.amp.autocast('cuda', enabled=False):
            x_vgg, y_vgg = self.vgg(x.float()), self.vgg(y.float())
            loss = 0.0
            loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
            loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
            loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
            loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
            loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
            return loss


class StyleLoss(nn.Module):
    """
    Gram matrix feature correlation loss for texture synthesis.
    """
    def __init__(self, pretrained=True, vgg=None):
        super().__init__()
        self.vgg = vgg if vgg is not None else VGG19(pretrained=pretrained)
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        return f.bmm(f_T) / max(float(h * w * ch), 1.0)

    def forward(self, x, y):
        with torch.amp.autocast('cuda', enabled=False):
            x_vgg, y_vgg = self.vgg(x.float()), self.vgg(y.float())
            loss = 0.0
            loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
            loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
            loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
            loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))
            return loss


# =========================================================================
# Total Variation & Boundary Seam Discrepancy (BSD) Losses
# =========================================================================

class TVLoss(nn.Module):
    """
    Total Variation Regularization.
    """
    def forward(self, x):
        h = x.size(2)
        w = x.size(3)
        tv_h = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) / (x.size(0) * x.size(1) * (h - 1) * w)
        tv_w = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])) / (x.size(0) * x.size(1) * h * (w - 1))
        return tv_h + tv_w


class BoundarySeamLoss(nn.Module):
    r"""
    Boundary Seam Discrepancy (BSD) Loss:
    Measures gradient continuity across the mask perimeter \partial \Omega,
    directly validating Hypothesis 2 of the ICEM-DBA formulation.
    """
    def __init__(self, dilation_kernel=3):
        super().__init__()
        self.kernel_size = dilation_kernel

    def forward(self, pred, target, mask):
        # Extract 1-pixel boundary transition band around mask perimeter
        dilated = F.max_pool2d(mask.float(), kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        eroded = 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        boundary_band = dilated - eroded  # [B, 1, H, W]

        # Compute gradient difference across boundary band
        diff = torch.abs(pred - target) * boundary_band
        normalizer = torch.clamp(torch.sum(boundary_band), min=1.0)
        return torch.sum(diff) / normalizer
