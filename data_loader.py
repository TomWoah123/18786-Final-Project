import glob
import os

import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import facenet_pytorch
import pandas as pd
import torch

feature_frame = pd.read_csv("ffhq_aging_labels.csv", index_col="image_number")


class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, main_dir, ext='*.png'):
        self.main_dir = main_dir
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        mtcnn = facenet_pytorch.MTCNN(image_size=256, margin=0)
        img_cropped = mtcnn(image)
        img_number = int(img_loc.split("_resized")[0][-5:])
        age_group = feature_frame.iloc[img_number].age_group
        age_group = 0 if age_group == "20-29" else 1  # Encode the age group to be 0 for 20-29 and 1 for 50-69
        age_vector = torch.zeros(2)
        age_vector[age_group] = 1
        return img_cropped, age_vector


def get_data_loaders(train_test_split=0.9, batch_size=16):
    twenties_female = CustomDataSet("data/20-29_female")
    train_size = int(twenties_female.__len__() * train_test_split)
    test_size = twenties_female.__len__() - train_size
    twenties_female_train, twenties_female_test = random_split(twenties_female, [train_size, test_size])
    twenties_male = CustomDataSet("data/20-29_male")
    train_size = int(twenties_male.__len__() * train_test_split)
    test_size = twenties_male.__len__() - train_size
    twenties_male_train, twenties_male_test = random_split(twenties_male, [train_size, test_size])
    fifties_female = CustomDataSet("data/50-69_female")
    train_size = int(fifties_female.__len__() * train_test_split)
    test_size = fifties_female.__len__() - train_size
    fifties_female_train, fifties_female_test = random_split(fifties_female, [train_size, test_size])
    fifties_male = CustomDataSet("data/50-69_male")
    train_size = int(fifties_male.__len__() * train_test_split)
    test_size = fifties_male.__len__() - train_size
    fifties_male_train, fifties_male_test = random_split(fifties_male, [train_size, test_size])
    combined_train = ConcatDataset([twenties_female_train, twenties_male_train, fifties_female_train, fifties_male_train])
    combined_test = ConcatDataset([twenties_female_test, twenties_male_test, fifties_female_test, fifties_male_test])
    train_dataloader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(combined_test, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader
