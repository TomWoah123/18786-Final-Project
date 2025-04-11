import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torchvision import transforms

IMAGE_SIZE = 512


class FGNetDataset(Dataset):

    def __init__(self, main_dir):
        self.main_dir = main_dir
        sub_directories = os.listdir(self.main_dir)
        data_points = []
        for d in sub_directories:
            files = os.listdir(os.path.join(self.main_dir, d))
            ages = []
            for f in files:
                age = f.split(".")[0]
                age = int(age)
                ages.append(age)
            n = len(ages)
            for i in range(n - 1):
                for j in range(i, n):
                    if abs(ages[i] - ages[j]) >= 18:
                        data_points.append((d, (ages[i], ages[j])))
                        data_points.append((d, (ages[j], ages[i])))
        self.data_points = data_points

    # def transform_type(trans_type = "basic"):
    #     # basic
    #     basic_transform = transforms.Compose([
    #         transforms.Resize(IMAGE_SIZE, Image.BICUBIC),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])

    #     # advanced
    #     load_size = int(1.25 * IMAGE_SIZE)
    #     osize = [load_size, load_size]
    #     advanced_transform = transforms.Compose([
    #         transforms.Resize(osize, Image.BICUBIC),
    #         transforms.RandomCrop(IMAGE_SIZE),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])

    #     if opts.data_preprocess == 'basic':
    #         train_transform = basic_transform
    #     elif opts.data_preprocess == 'advanced':
    #         # todo: add your code here: below are some ideas for your reference
    #         train_transform = advanced_transform        
    #         # pass

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        person_number, (current_age, target_age) = self.data_points[idx]
        current_age_image_path = os.path.join(self.main_dir, person_number, f"{current_age:03}.png")
        target_age_image_path = os.path.join(self.main_dir, person_number, f"{target_age:03}.png")
        # input_image = Image.open(current_age_image_path).convert("RGB")
        # target_image = Image.open(target_age_image_path).convert("RGB")
        
        input_image = Image.open(current_age_image_path)
        target_image = Image.open(target_age_image_path)

        transform = transforms.ToTensor()
        input_tensor = transform(input_image)
        output_tensor = transform(target_image)
        current_age_map = torch.full(size=(1, 512, 512), fill_value=current_age / 69)
        target_age_map = torch.full(size=(1, 512, 512), fill_value=target_age / 69)
        input_tensor = torch.cat([input_tensor, current_age_map, target_age_map], dim=0)
        return input_tensor, output_tensor


def get_data_loaders(opts, batch_size=16, test_split=0.9):
    fgnet_dataset = FGNetDataset("organized_images")
    train_size = int(fgnet_dataset.__len__() * test_split)
    test_size = fgnet_dataset.__len__() - train_size
    fgnet_train, fgnet_test = random_split(fgnet_dataset, [train_size, test_size])
    fgnet_train_dataloader = DataLoader(fgnet_train, batch_size=batch_size, shuffle=True)
    fgnet_test_dataloader = DataLoader(fgnet_test, batch_size=batch_size, shuffle=True)
    return fgnet_train_dataloader, fgnet_test_dataloader


