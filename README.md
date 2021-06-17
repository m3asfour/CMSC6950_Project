# CMSC6950_Project

## PyAutoLens Project
#### PyAutoLens is a python package for gravitational lens problems, which can be summarized by the bending of the light travelling across the galaxy. 
The light can bend due to the mass of nearby galaxies or even black holes, lens galaxies, which prohibits us from reaching a clear visualization from distant stars past our solar system. 
<br><br>
This porject uses PyAutoLens to generate synthesized images of the gravitational lens phenomenon, and trains a convolutional neural network (CNN) to train on raw images from space telescopes to predict properties of the distant starts, source galaxies, to better understand them from their bent light.
<br>


##### ![](https://sites.astro.caltech.edu/~george/qsolens/lensillustration.jpg) Image credits: https://sites.astro.caltech.edu/~george/qsolens/
---

## 1. Create Conda Environment and Install Required Packages
```
conda create -n pyautolens python==3.9
conda activate pyautolens
pip install autoconf numpy pandas matplotlib tensorflow keras tqdm
```
<br>

## 2. Build PyAutoLens from source
### First make sure you are in the project's directory.
### 2.1 Clone the PyAutoLens repo and its supporting modules
#### 2.1.1 You can either clone them manually using the following commands
```
mkdir packages
cd packages
git clone https://github.com/rhayes777/PyAutoFit
git clone https://github.com/Jammy2211/PyAutoArray
git clone https://github.com/Jammy2211/PyAutoGalaxy
git clone https://github.com/Jammy2211/PyAutoLens
cd ..
```
#### 2.1.2 or using the Makefile command, and it will take create directory and clone the repos
```
make packages
```
* #### You can also delete the packages using
    ```
    make clean_packages
    ```
<br>

### 2.2 Install dependencies
```
conda activate pyautolens
pip install -r packages/PyAutoFit/requirements.txt
pip install -r packages/PyAutoArray/requirements.txt
pip install -r packages/PyAutoGalaxy/requirements.txt
pip install -r packages/PyAutoLens/requirements.txt
```
<br>

### 2.3 Add repos to the environment path
```
conda-develop packages/PyAutoFit
conda-develop packages/PyAutoArray
conda-develop packages/PyAutoGalaxy
conda-develop packages/PyAutoLens
```

## 3. Build from Makefile
```
make
```
* You can also delete all temporary files using
    ```
    make clean
    ```
* or delete all generated files (temporary, figures, report) except for the packages using
  ```
  make clean_all
  ```