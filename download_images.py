import json
import requests
import os
from PIL import Image
import pandas as pd
import random

age_range = input("Please input an age range from the FFHQ Dataset: ")
valid_age_ranges = ["0-2", "3-6", "7-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-69", "70-120"]
if age_range not in valid_age_ranges:
    raise Exception(f"Invalid age range! Please only list from these age ranges: {valid_age_ranges}")

gender = input("Please input a gender from the FFHQ Dataset: ")
if gender.lower() != "male" and gender.lower() != "female":
    raise Exception(f"Invalid gender! Please only specify male or female")

category_directory = os.path.join(".", f"{age_range}_{gender.lower()}")
if not os.path.exists(category_directory):
    os.mkdir(category_directory)

feature_frame = pd.read_csv("ffhq_aging_labels.csv")
important_features = feature_frame[["image_number", "age_group", "gender"]]
grouped_features_dictionary = dict(important_features.groupby(["age_group", "gender"])["image_number"].groups)
image_numbers = grouped_features_dictionary[(age_range, gender.lower())]
random.seed(42)
image_numbers_subset = random.sample(list(image_numbers), k=500)
print(image_numbers_subset)

with open("ffhq-dataset-v2.json", "r") as file:
    data = json.load(file)
    for image_number in image_numbers_subset:
        google_url = data[str(image_number)]["image"]["file_url"]
        try:
            response = requests.get(google_url, stream=True)
            response.raise_for_status()
            file_size = int(response.headers.get('content-length', 0))

            filepath = os.path.join(".", "test_file.png")

            with open(filepath, 'wb') as f:
                downloaded_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    print(f"Downloaded {downloaded_size} / {file_size} bytes")
            print(f"File downloaded successfully to {filepath}")
            im = Image.open(filepath)
            imResize = im.resize((256, 256), Image.ANTIALIAS)
            padded_number = f"{image_number:05}"
            imResize.save(f"{age_range}_{gender.lower()}/{padded_number}_resized.png", "PNG")
            os.remove(filepath)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
        except IOError as e:
            print(f"IOError: {e}")
