[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/6ndC2138)


# CS 175 Final Project 

## Artificial Unintelligence

Group Members: AJ Moon,
Brian Le,
Joshua Cordero,
Noah Risser


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Weights and Biases](https://img.shields.io/badge/Weights%20&%20Biases-FFBE00.svg?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Overview

This project's goal is to use machine learning in order to analyze areas with settlements and/or electricity using Sentinel 2 and VIIRS satellite images. We employ a Self-Supervised Deep Learning model to semantically segment satellite images. We used SimCLR, which is a self-supervised model used to generate our initial weights for our Deep Learning model, U-Net++. This repo contains scripts to run our code, our various models including various supervised models, test cases, and our best model weights trained on these images.

Project documents
[Presentation](https://docs.google.com/presentation/d/1oqYLN1-L_TKw-rn5uoV_nzOL2YuNyasb_uZdeHgEXL4/edit?usp=drive_link)
[Poster](https://docs.google.com/presentation/d/1cKBhTp4_c819uSwylguZbLjDkGQST18T60RmnO3qsj4/edit?usp=drive_link)
[Tech Memo](https://docs.google.com/document/d/18lNjLlPdIC-aW2yqYVh0Pntn_LyG24932s1Ss93IPhM/edit?usp=drive_link)

## Installation
The Code requires `python>3.10` as well `pip` package manager.

Clone the repo

```
git clone git@github.com:cs175cv-s2024/final-project-artificial-unintelligence.git

cd final-project-artificial-unintelligence
```

Create virtual environment and activate it

```
python -m venv env
source ./env/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

## Run Commands
Open [utilities.py](./src/utilities.py) and set preferred model.

To train a supervised model run
```
python scripts/supervised/train.py
```
To train a self-supervised model run
```
python scripts/unsupervised/train.py
```

To evaluate these models run
```
python scripts/supervised/evaluate.py --model_path=<path/to/model.cpkt>
```

Pretrained models can be found under the models/ folder.


## ML Pipeline
![ML Pipeline Diagram](assets/ML_Pipeline.png)

## SimCLR
SimClr is a self-supervised deep learning architecture that utilizes image augmentations to learn image representations. The primary augmentation used are vertical flip, horizontal flip, and rotation by 90 degrees. The augmentations allow the same unlabeled image to be fed into the backbone encoder twice with their outputs from the projection head to be compared. The goal of this model is to minimize the difference between the output of the same image with augmentations performed on it. 
![SimClr Architecture](https://amitness.com/posts/images/simclr-general-architecture.png)

The weights learned for this model can then be used for the downstream task of image segmentation by attaching a different segmentation head. The decoders available for use with our model are fcn_resnet, deeplabv3, and unet++. 

## U-Net++
U-Net++ is a deep-learning model used for semantic segmentation tasks. It builds upon a U-Net network by adding nested dense skip connections. The model first encodes the images using convolution and pooling layers by reducing the size and increasing the amount of features the model uses to learn.  When decoding and upscaling the images back to their initial resolution, it utilizes nested dense skip connections in order to preserve more spatial information. 

![Unet++ Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20230628132335/UNET.webp)

## Sample Results
Below are some sample results. Areas of interest are Human Settlements without electricity which are the red pixels. Our models also classify No Human Settlements without electricity (Blue), Human Settlements with electricity (Yellow), and No Human Settlements with electricity (Purple)

![Sample Result Images](assets/Sat_Img_Sample.png)

## SimCLR Training
Scatterplot of the outputs of SimCLR encodings

![SimCLR Scatterplot](assets/SimCLR_Scatterplot.png)

Nearest Neighbor Plots of SimCLR

![Nearest Neighbor Plot 3](assets/NN_Plot3.png)
![Nearest Neighbor Plot 31](assets/NN_Plot31.png)

## Citing Artificial Unintelligence CS 175 Final Project 
```
@misc{Cordero2024Artificial,
  title={Artificial Unintelligence CS 175 Final Project},
  author={Joshua Cordero, Brian Le, AJ Moon, Noah Risser},
  year={2024}
}
```
