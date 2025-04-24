import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random


class UTKDataset(Dataset):
    def __init__(self, age_group, directory="utk_dataset", num_samples=5200, load_statistics=False):
        random.seed(42)
        age_dictionary = {"0-9": [], "10-19": [], "20-29": [], "30-39": [], "40-49": [], "50-69": [], "70+": []}
        assert age_group in age_dictionary
        
        # Create a dictionary to help get the statistics for the dataset
        # Making each array bigger than the actual size to account for the "unknowns"
        self.data_set_statistics = {"age":[0 for i in range(117)], 
                               "gender":[0 for i in range(3)], 
                               "race":[0 for i in range(6)],
                               "total": 0}
        

        for direc in os.listdir(directory):
            # This is a mac problem.... there is a file called .ds_store that trips this pipeline up
            if direc == ".DS_Store":
                continue
            direc_path = os.path.join(directory, direc)
            for file in os.listdir(direc_path):

                age = int(file.split("_")[0])
                gender = int(file.split("_")[1])
                race = file.split("_")
                
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
              
                self.data_set_statistics["age"][age - 1] += 1 # Increment the number at the age (saying "age - 1" because the list is 0 indexed and age starts at 1)
                self.data_set_statistics["gender"][gender] += 1 # Increment the gender (the gender is 0 for male and 1 for women)
                
                if len(race) > 1: 
                    self.data_set_statistics["race"][-1] += 1 # Increment the race (according the the data website: "race is integer from 0 to 4, denoting White, Black, Asian, Indian, and Others")
                else:
                    race = int(race)
                    self.data_set_statistics["race"][race] += 1

                self.data_set_statistics["total"] += 1

        self.data_points = random.sample(age_dictionary[age_group], num_samples)

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        image_path = self.data_points[idx]
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128), Image.BICUBIC),
            transforms.CenterCrop((128, 128)),
            # transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        image_tensor = transform(image)
        return image_tensor
    
    def get_entire_utk_data_statistics(self):
        print("------------- UTK Statistics -------------")
        print("\t* Age:")
        print(f"\t{' ' * 4}0-9: {sum(self.data_set_statistics["age"][:9])} ({(sum(self.data_set_statistics["age"][:9])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}10-19: {sum(self.data_set_statistics["age"][9:20])} ({(sum(self.data_set_statistics["age"][9:20])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}20-29: {sum(self.data_set_statistics["age"][20:29])} ({(sum(self.data_set_statistics["age"][20:29])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}30-39: {sum(self.data_set_statistics["age"][30:99])} ({(sum(self.data_set_statistics["age"][30:39])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}40-49: {sum(self.data_set_statistics["age"][40:49])} ({(sum(self.data_set_statistics["age"][40:49])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}50-59: {sum(self.data_set_statistics["age"][50:59])} ({(sum(self.data_set_statistics["age"][50:59])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}60-69: {sum(self.data_set_statistics["age"][60:69])} ({(sum(self.data_set_statistics["age"][60:69])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}70+: {sum(self.data_set_statistics["age"][70:-1])} ({(sum(self.data_set_statistics["age"][70:-1])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}UNKNOWN: {sum(self.data_set_statistics["age"][-1])} ({(sum(self.data_set_statistics["age"][-1])/self.data_set_statistics["total"])*100}%)")
        print("- - - -")
        print("\t* Gender:")
        print(f"\t{' ' * 4}Male: {sum(self.data_set_statistics["gender"][0]) } ({(sum(self.data_set_statistics["gender"][0])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}Female: {sum(self.data_set_statistics["gender"][1])} ({(sum(self.data_set_statistics["gender"][1])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}UNKNOWN: {sum(self.data_set_statistics["gender"][2])} ({(sum(self.data_set_statistics["gender"][2])/self.data_set_statistics["total"])*100}%)")
        print("- - - -")
        print("\t* Race:")
        print(f"\t{' ' * 4}White: {sum(self.data_set_statistics["race"][0]) } ({(sum(self.data_set_statistics["race"][0])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}Black: {sum(self.data_set_statistics["race"][1])} ({(sum(self.data_set_statistics["race"][1])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}Asian: {sum(self.data_set_statistics["race"][2]) } ({(sum(self.data_set_statistics["race"][2])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}Indian: {sum(self.data_set_statistics["race"][3])} ({(sum(self.data_set_statistics["race"][3])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}Others: {sum(self.data_set_statistics["race"][4])} ({(sum(self.data_set_statistics["race"][4])/self.data_set_statistics["total"])*100}%)")
        print(f"\t{' ' * 4}UNKNOWN: {sum(self.data_set_statistics["race"][5])} ({(sum(self.data_set_statistics["race"][5])/self.data_set_statistics["total"])*100}%)")
        




def get_both_data_loaders(young_age_group, old_age_group, batch_size=16):
    young_utk_dataset = UTKDataset(age_group=young_age_group)
    old_utk_dataset = UTKDataset(age_group=old_age_group)

    young_dataloader = DataLoader(young_utk_dataset, batch_size=batch_size, shuffle=True)
    old_dataloader = DataLoader(old_utk_dataset, batch_size=batch_size, shuffle=True)

    both_ages_dataloader = zip(young_dataloader, old_dataloader)
    
    return both_ages_dataloader

def get_data_loaders(age_group, batch_size=16):
    utk_dataset = UTKDataset(age_group=age_group)
    dataloader = DataLoader(utk_dataset, batch_size=batch_size, shuffle=True)
    return dataloader




