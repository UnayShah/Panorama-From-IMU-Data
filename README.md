
# ECE 276A: Orientation Mapping and Panorama Creation

## Introduction
This is project 1 of the course [ECE 276A: Sensing & Estimation in Robotics](https://natanaso.github.io/ece276a/) at UCSD, being taught by professor [Nikolay Atanisov](https://natanaso.github.io/).

This project provides IMU, VICON and camera data. Using these, it is required to improve reading of the angular velocity component of IMU data using gradient descent with the linear acceleration as the ground truth. This data can be used to find the orientation of the sensor setup in terms of [Euler angles](https://natanaso.github.io/ece276a/ref/ECE276A_3_Rotations.pdf#page=11) which can give the Rotation Matrix of the image. Using the rotation matrices of the images, they can be stitched into a panorama.

## Running the code
1. Download the dataset from [this Google Drive link](https://drive.google.com/drive/folders/1-Ot6Yr_zCEMKgYN5hIXYfysUy09qx_G3)
2. Place both the datasets in the same folder with the following file structure:

        .
        ├── testset
        │   ├── cam
        │   └── imu
        └── trainset
        ├── cam
        ├── imu
        └── vicon
3. Install the following packages (assuming numpy and matplotlib are already installed):

        pip install jax
        pip install jaxlib
        pip install transforms3d

4. Run the file project_1.py

        python project_1.py
5. When prompted, enter the folder path where __both__ traindata and testdata folders are located.
6. When prompted, enter the dataset number to run the code on. __Enter an integer from 1 to 11__.

## Miscellaneous Information
The datasets are labelled 1-11. Datasets 1, 2, 8, 9, 10, 11 have camera data, so they will create 2 types of panoramas: [Lambert azimuthal equal-area projection](https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection) and [Cylindrical projection from Spherical World coordinates](https://natanaso.github.io/ece276a/ref/ECE276A_5_LocalizationOdometry.pdf#page=13).

All datasets will generate 2 sets of graphs showing the Euler angles and linear acceleration:
    
+ ### Angular velocity is compared between:
    - IMU data which is unprocessed (only bias corrected)
    - IMU data after gradient descent
    - VICON data

    Quaternions and rotation matrices are converted to Euler angles using *transform3d* package
    
+ ### Linear acceleration is compared between:
    - Values obtained from angular velocity (IMU data only bias corrected)
    - Values obtained from angular velocity after gradient descent
    - IMU data which is ground truth (bias corrected)
    
## Results
Here are the results obtained from dataset 8:

__Euler angle comparison for dataset 8:__

![Euler angle comparison for dataset 8](https://github.com/UnayShah/ece276A_pr1/blob/master/plots/8_W.jpg)

__Linear acceleration comparison for dataset 8__

![Linear acceleration comparison for dataset 8](https://github.com/UnayShah/ece276A_pr1/blob/master/plots/8_A.jpg)

__Lambert Azimuthal panorama for dataset 8__

![Lambert Azimuthal panorama for dataset 8](https://github.com/UnayShah/ece276A_pr1/blob/master/plots/panorama_lamber_8.jpg)

__Cylindrical panorama for dataset 8__

![Cylindrical panorama for dataset 8](https://github.com/UnayShah/ece276A_pr1/blob/master/plots/panorama_8.jpg)