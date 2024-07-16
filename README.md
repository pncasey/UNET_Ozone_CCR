# UNET Model Training for Ozone Soundings

Author: Philip Casey (pcasey@stcnet.com)

This repository details how to train a UNET style neural network to retrieve Ozone sounding information from the Cross-track Infrared Sounder (CrIS) instrument aboard NOAA-20. The python scripts included show how to process the data and train the model. 

## Getting Started

The data used as inputs to the UNET model comes the NOAA Open Data Dissemination (NODD) Program. Amazon Web Services hosts open NOAA and which can be transferred to your machine or cloud infrastructure. This particular project looks at the CrIS Radiance measurements after the NUCAPS Cloud Clearing Radiance (CCR) algorithm has been applied. You can browse those files here https://noaa-jpss.s3.amazonaws.com/index.html#NOAA20/SOUNDINGS/NOAA20_NUCAPS-CCR/. The UNET model is trained against MERRA-2 reanalysis data 

## Scripts

* CCR_Gridding.py: Grids the relevant NUCAPS CCR radiances and satellite geometries to a .5 degree spatial resolution. 
* Data preprocessing.py: Makes both the input and target data machine learning ready by filling missing values and matching the data spatially. Each day's data is saved as a .npy file for easy loading by the keras/tensorflow data loader.
* UNET_training.py: Creates a custom keras data generator for to call each batch of input and target data. Defines the net work architecture of the model. Trains and saves the trained model.
