import os
import shutil

utk_folder = '../UTKFace-20250416T172452Z-001/UTKFace'
young_folder = '../young_folder'
old_folder = '../old_folder'

for img_name in os.listdir(utk_folder):

    # Image path 
    img_path = os.path.join(utk_folder, img_name)

    # Understand the metadata of the image
    metadata = img_name.split("_")
    age = int(metadata[0])

    if age >= 20 and age <= 29:
        shutil.copy(img_path, '../young_folder')
    elif age >= 50 and age <= 69:
        shutil.copy(img_path, '../old_folder')