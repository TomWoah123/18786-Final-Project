import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

image_directory = "organized_images"
if not os.path.exists(image_directory):
    os.mkdir(image_directory)

transform = transforms.Compose([
    transforms.Resize((512, 512), Image.BICUBIC),
    transforms.ToTensor()
])

store_as_grayscale = False

if store_as_grayscale:
    print("Storing the image as grayscale")
else:
    print("Storing the data as RGB")

print("Starting")
for file in os.listdir("FGNET/images"):
    # Want to check if this is a file (and not the .DS_Store directory)
    if not os.path.isfile(os.path.join("FGNET/images", file)):
        continue

    # print(file)
    name_components = file.split("A")
    person_number = name_components[0]
    person_path = os.path.join(image_directory, person_number)
    if not os.path.exists(person_path):
        os.mkdir(person_path)
    if store_as_grayscale:
        image = Image.open(os.path.join("FGNET/images", file)).convert("L")
    else:
        image = Image.open(os.path.join("FGNET/images", file))

    tensor_image = transform(image)
    age = name_components[1].split(".")[0]
    if len(age) == 3:
        age = age[:2]
    age = int(age)
    image_path = os.path.join(person_path, f"{age:03}.png")
    save_image(tensor_image, image_path)



print("Done")