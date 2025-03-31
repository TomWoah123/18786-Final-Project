<div align="center">
<h1>Exploring Generative Adversarial Networks for Re-Aging Faces</h1>

</div>
<div align="center">
<b>Timothy Wu</b><sup>1*</sup>,
<b>Penny Sarac</b><sup>1*</sup>,
<b>Briana Abam</b><sup>1*</sup>
<br>
</div>
<div align="center">
<sup>1</sup>Carnegie Mellon University
</div>

## Environment Set Up
To set up the current environment, please clone the repository

HTTPS URL
```bash
cd <Project Folder>
git clone https://github.com/TomWoah123/18786-Final-Project.git
```
or 

SSH URL
```bash
cd <Project Folder>
git clone git@github.com:TomWoah123/18786-Final-Project.git
```

After you have cloned the repository, you will need to download two specific files:
- `ffhq-dataset-v2.json`: This JSON file holds information regarding each of the 70,000 images in the dataset,
specifically where to download the image. The JSON file can be found in this 
[Google Drive Link](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP). If you are having
problems accessing the link, feel free to reach out to Timothy Wu (tpwu@andrew.cmu.edu) and he will provide you the
file.
- `ffhq_aging_labels.csv`: This CSV file contains the data labels for each image, specifically the gender and age range
that the person in the image falls under. The CSV file can be found and downloaded from the original GitHub link located
[Here](https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/ffhq_aging_labels.csv)

Make sure that both files are in the same directory as the `download_images.py` file for you to properly download the
images. Your folder should look something like this:
```
project
|   .gitignore
│   README.md
│   download_images.py
|   ffhq-dataset-v2.json
|   ffhq_aging_labels.csv
└───20-29_female
│   │   XXXXX.png
│   │   XXXXX.png
│   │   ...
└───20-29_male
    │   XXXXX.png
    │   XXXXX.png
```
Ignore the directories for the images for now; they will be created once you run the `download_images.py` script.

## Downloading Images
Downloading images should be as simple as running the `download_images.py` script which will ask you for the age-range
and gender. The script will then randomly select 500 images from the dataset that fit the age range and gender
specifications and download them. The script also resizes the images from the original 1024 x 1024 size down to
256 x 256 to save space since we don't have 800 GPUs. The script may take some time (~20ish minutes to run for one
age range and gender).
