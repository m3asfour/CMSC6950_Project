# CMSC6950_Project

## PyAutoLens Project
#### This project uses the PyAutoLens modules and its supporting ones to generate adversial examples of strong gravitational lenses. Strong gravitational lenses occur when light from far away stars or galaxies bind due to passing by bodies with high graviation, known as lens galaxies/bodies. This results into recieving morphed shapes of the galaxies' light. Such adversial examples are to be used to train a convolutional neural network for predicting the properties of source and lens galaxies, such as mass. 

##### ![](https://sites.astro.caltech.edu/~george/qsolens/lensillustration.jpg) Image credits: https://sites.astro.caltech.edu/~george/qsolens/
---

## 1. Build PyAutoLens from source
### 1.1 Clone the PyAutoLens with supporting repos
```
git clone https://github.com/rhayes777/PyAutoFit
git clone https://github.com/Jammy2211/PyAutoArray
git clone https://github.com/Jammy2211/PyAutoGalaxy
git clone https://github.com/Jammy2211/PyAutoLens
```
### 1.2 Create Python environment 
```
conda create -n pyautolens python==3.9
conda activate pyautolens
```
### 1.3 Install dependencies
```
pip install -r PyAutoFit/requirements_source.txt
pip install -r PyAutoArray/requirements_source.txt
pip install -r PyAutoGalaxy/requirements_source.txt
pip install -r PyAutoLens/requirements_source.txt
pip install autoconf
```
### 1.4 Add repos to the environment path
```
conda-develop PyAutoFit
conda-develop PyAutoArray
conda-develop PyAutoGalaxy
conda-develop PyAutoLens
```

