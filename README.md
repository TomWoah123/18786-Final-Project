<div align="center">
<h1>Exploring Generative Adversarial Networks for Re-Aging Faces</h1>

</div>
<div align="center">
<b>Timothy Wu</b><sup>1*</sup>,
<b>Pelinsu Sarac</b><sup>1*</sup>,
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

Next, you will need to download the UTK dataset which can be found using this link: [https://www.kaggle.com/datasets/moritzm00/utkface-cropped](https://www.kaggle.com/datasets/moritzm00/utkface-cropped). You will need to unzip the file and leave the image folders in the root directory of your project. Finally, you will need to install of the packages needed.
```bash
pip install -r requirements.txt
```

## Structural Overview
A preview of the project structure can be found below.
```
.
├── Calculate_FID_and_KID.ipynb
├── data_loader.py
├── requirements.txt
├── README.md
├── savvides.png
├── streamlit_runner_page.py
├── train_cycle_gan.py
├── zanette.png
├── utk_models.py
└── utk_dataset.py
```
A description of each python file can be found below:
- `streamlit_runner_page.py`: This python file hosts the front end user interface for how a user can interact with our model.
- `train_cycle_gan.py`: This python file contains the training script for training the CycleGAN architecture.
- `utk_dataset.py`: This python file hosts the Dataset and DataLoader classes used to load in the faces of a particular age range in the UTK dataset.
- `utk_models.py`: This python file contains the architectures for the Deep Convolutional Discriminator (DC-Discriminator) and Deep Convolutional Generator (DC-Generator).

