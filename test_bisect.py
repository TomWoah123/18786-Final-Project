import torch
import torchmetrics.image as tm

perceptual_loss = tm.lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg")
generated_images = torch.rand(16, 3, 512, 512) * 2 - 1
target_images = torch.rand(16, 3, 512, 512) * 2 - 1
result = perceptual_loss(generated_images, target_images)
print(result)

