# CMSC6950_Project

## PyAutoLens Project
#### PyAutoLens is a python package for gravitational lens problems, which can be summarized by the bending of the light travelling across the galaxy. 
The light can bend due to the mass of nearby galaxies or even black holes, lens galaxies, which prohibits us from reaching a clear visualization from distant stars past our solar system. 
<br><br>
This porject uses PyAutoLens to generate synthesized images of the gravitational lens phenomenon, and trains a convolutional neural network (CNN) to train on raw images from space telescopes to predict properties of the distant starts, source galaxies, to better understand them from their bent light.
<br>


##### ![](https://sites.astro.caltech.edu/~george/qsolens/lensillustration.jpg) Image credits: https://sites.astro.caltech.edu/~george/qsolens/
---

## 1. Installing with Conda
### First make sure you are in the project's directory.
### 1.1 Create the conda environment with required dependencies
```
conda create -n autolens astropy numba numpy scikit-image scikit-learn scipy
```
<br>

### 1.2 Install other dependencies
```
conda activate autolens
pip install --upgrade pip
pip install autolens --ignore-installed numba llvmlite
```
<br>


## 2. Build from Makefile
```
make
```
* You can also delete all temporary files using
    ```
    make clean
    ```
* or delete all generated files (temporary, figures, report)
  ```
  make reset
  ```

### The report is now generated in the project's directory with a CNN model as a .h5 file