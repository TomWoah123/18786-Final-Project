import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class UTKDataset(Dataset):
    def __init__(self, age_group, directory="utk_dataset"):
        age_dictionary = {"0-9": [], "10-19": [], "20-29": [], "30-39": [], "40-49": [], "50-69": [], "70+": []}
        assert age_group in age_dictionary
        for direc in os.listdir(directory):
            direc_path = os.path.join(directory, direc)
            for file in os.listdir(direc_path):
                age = int(file.split("_")[0])
                if age >= 70:
                    age_dictionary["70+"].append(os.path.join(direc_path, file))
                elif age >= 50:
                    age_dictionary["50-69"].append(os.path.join(direc_path, file))
                elif age >= 40:
                    age_dictionary["40-49"].append(os.path.join(direc_path, file))
                elif age >= 30:
                    age_dictionary["30-39"].append(os.path.join(direc_path, file))
                elif age >= 20:
                    age_dictionary["20-29"].append(os.path.join(direc_path, file))
                elif age >= 10:
                    age_dictionary["10-19"].append(os.path.join(direc_path, file))
                else:
                    age_dictionary["0-9"].append(os.path.join(direc_path, file))
        self.data_points = age_dictionary[age_group]

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        image_path = self.data_points[idx]
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128), Image.BICUBIC),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform(image)
        return image_tensor


def get_data_loaders(age_group, batch_size=16):
    utk_dataset = UTKDataset(age_group=age_group)
    dataloader = DataLoader(utk_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

