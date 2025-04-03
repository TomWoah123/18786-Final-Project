import torch
from PIL import Image
from torchvision import transforms
from bisenet_model import BiSeNet
import matplotlib.pyplot as plt

bisenet_model = BiSeNet(19)
bisenet_model.load_state_dict(torch.load("79999_iter.pth", map_location=torch.device("cpu")))
image = Image.open("organized_images/001/008.png")
image_two = Image.open("organized_images/001/028.png")
transform = transforms.ToTensor()
image_tensor = transform(image)
image_two_tensor = transform(image_two)
data = torch.stack([image_tensor, image_two_tensor], dim=0)
print(data.shape)
with torch.no_grad():
    result, _, _ = bisenet_model(data)
print(result[0, 0])
plt.imshow(result[1, 1])
plt.show()

