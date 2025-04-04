from torch import nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children())[:9])  # Extract layers up to relu4_1
        for param in self.features.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, generated_image, target_image):
        loss = nn.MSELoss()
        generated_image = F.interpolate(generated_image, size=224)
        target_image = F.interpolate(target_image, size=224)
        gen_features = self.features(generated_image)
        target_features = self.features(target_image)
        return loss(gen_features, target_features)
