import sys
import math
import xarray as xr
from netCDF4 import Dataset
import scipy.interpolate
import timeit
from datetime import datetime, timedelta
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy

import datetime as dt
from pyresample.geometry import SwathDefinition
from pyresample import geometry
from pyresample.kd_tree import resample_nearest


import optuna.integration.lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import log_evaluation
import sklearn.datasets
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit

from sklearn.impute import KNNImputer
from skimage.transform import resize
import tensorflow as tf

nloschmidt = 2.6867775e+19  # NIST Loschmidt's molec/cm^3 per atm @ stdP
t_std = 273.15  # standard temperature [K]
mw_oz = 47.9982  # gm/mole ozone
navog = 6.02214199e+23
mw_d = 28.9644  # gm/mole dry air
g_std = 980.664  # acceleration of gravity, cm/s^2
mw_oz = 47.9982
p_std = 1013.25
oz_eps = mw_oz / mw_d
cdair = 1000.0 * p_std * navog / (mw_d * g_std)
FILL_VAL = np.nan


def convert_mr2cd(O3):
    pres = MERRA_pressure
    nlev = len(pres)
    MERRA_ozcd = np.ones((8, 72, 361, 576), dtype=float) * FILL_VAL
    delta_p = np.zeros((nlev), dtype=float)
    delta_p[0] = pres[0]
    delta_p[1:nlev - 1] = pres[1:nlev - 1] - pres[0: nlev - 2]
    for i in range(8):
        for j in range(361):
            for k in range(576):
                ozmr = O3[i,:,j,k]
                # ozmr = ozmr*(1e9*(28.6/48))
                for l in range(nlev):
                    MERRA_ozcd[i,l,j,k] = ozmr[l] * ((cdair * delta_p[l] / p_std) / oz_eps)
    return MERRA_ozcd

def calc_tot(ozcd):
    botlev = np.ones((8, 361, 576), dtype=float) * 71
    blmult = np.ones((8, 361, 576), dtype=float) * 1
    shape = np.shape(ozcd)
    nlev = shape[1]
    time = shape[0]
    lat = shape [2]
    lon = shape [3]

    totoz = np.ones((8, 361, 576), dtype=float) * FILL_VAL

    for i in range(time):
            for j in range(lat):
                 for k in range(lon):
                    totoz[i,j,k] = ozcd[i, 0 , j, k]

                    if botlev[i,j,k] < nlev - 1:
                        sfc = int(botlev[i,j,k])
                    if botlev[i,j,k] == nlev - 1:
                        sfc = nlev - 1

                    for l in range(1, int(sfc)):

                        if math.isnan(ozcd[i,l,j,k]) == False:
                            totoz[i,j,k] = totoz[i,j,k] + ozcd[i,l,j,k]

                    totoz[i,j,k] = totoz[i,j,k] + ozcd[i, int(71),j,k]
                    totoz[i,j,k] = totoz[i,j,k] * 1000.0 / nloschmidt
    return totoz

def ResampleToNUCAPSgrid(X, Y, data, newX, newY):
    swath_def1 = geometry.SwathDefinition(lons=X, lats=Y)
    swath_def2 = geometry.SwathDefinition(lons=newX, lats=newY)
    Resampled_Data = resample_nearest(swath_def1, data, swath_def2, radius_of_influence=50000, epsilon=1, fill_value=None)
    return Resampled_Data

def impute_target(array):
    imputedata = np.empty((360,720,1))
    imputer = KNNImputer(n_neighbors=10)
    A = imputer.fit_transform(y_var[:,:])
    imputedata[:,:,0] = A

    return imputedata

def impute_input(array):
    imputedata = np.empty((360,720,133))
    imputer = KNNImputer(n_neighbors=10)
    for j in np.arange(0,133,1):
        A = imputer.fit_transform(array[:,:,j])
        imputedata[:,:,j] = A

    return imputedata
# -----------------------------

MERRA_list = sorted(glob.glob('/mnt/nucaps-s3/philip/MERRA-2/2022-23/'+'*.nc4'))
NUCAPS_list = sorted(glob.glob('/mnt/nucaps-s3/philip/RadianceNUCAPS/2022-2023/'+'Ozone_CCR_Temp_Gridded*'))

# index = np.arange(0,10,1)
ascend = np.arange(0,2,1)

# index = np.arange(0,31,1)

# index = np.arange(0,10,1)
index = np.arange(91,95,1)
# index

MERRA_pressure = np.asarray( [0.01,0.02,0.0327,0.0476,0.066,0.0893,0.1197,0.1595,0.2113,0.2785,0.365,0.4758,0.6168,0.7951,1.0194,1.3005,1.6508,2.085,2.6202,3.2764,4.0766,5.0468,6.2168,7.6198,9.2929,11.2769,13.6434,16.4571,19.7916,23.7304,28.3678,33.81,40.1754,47.6439,
            56.3879, 66.6034, 78.5123,92.3657,108.663,127.837,150.393,176.93,208.152,244.875,288.083,337.5,375,412.5,450,487.5,525,562.5,600,637.5,675,700,725,750,775,800,820,835,850,865,880,895,910,925,940,955,970,985])
nlevels = np.arange(0,72,1)

fulldata = np.empty((0,133))
ylabel = np.empty((0))

for iteration in index:

    nucaps1 = nc.Dataset(NUCAPS_list[iteration])


    ## Adding Radiance data

    rad = nucaps1.variables['CrIS_Radiances'][:,:,0,:]
    rad = rad.reshape((259200,130))

    ##

    ##Adding Solar Zenith Angle
    solar_zenith = nucaps1.variables['Solar_Zenith'][:,:,0].flatten()
    view_angle = nucaps1.variables['View_Angle'][:,:,0].flatten()
    satellite_height = nucaps1.variables['Satellite_Height'][:,:,0].flatten()

    print('done with day', iteration)


    # rad_full = np.append(rad_full, rad, axis=0)
    # solar_zenith_full  = np.append(solar_zenith_full, solar_zenith, axis=0)

    ##Read in MERRA-2 data, extract latitude longitude and ozone data
    MERRAfile = nc.Dataset(MERRA_list[iteration])
    lons = MERRAfile.variables['lon'][:]
    lats = MERRAfile.variables['lat'][:]
    O3 = MERRAfile.variables['O3'][:][:][:][:]

    MERRA_ozcd = convert_mr2cd(O3)
    totoz = calc_tot(MERRA_ozcd)

    ##Read in NUCAPS Data with MERRA-2 Time designation
    nucaps1 = nc.Dataset(NUCAPS_list[iteration])
    nuMERRAtime = nucaps1.variables['MERRA-2_Time'][:,:,:]
    nuMERRAtime = nuMERRAtime.filled(np.nan)
    nulons = nucaps1.variables['lon'][:]
    nulats = nucaps1.variables['lat'][:]


    #MERRA-2 codes latitude and longitude in 1d. Both need to be in 2d to use pyresample. 
    full_lons = np.reshape(lons, (576,1))
    full_lons = np.repeat(full_lons.T, repeats = 361, axis = 0)

    full_lats = np.reshape(lats, (361,1))
    full_lats = np.repeat(full_lats, repeats = 576, axis = 1)

    #Regrid each time slice to nucaps gridding

    m0 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[0][:][:], nulons, nulats)
    m0 = m0.filled(np.nan)
    m1 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[1][:][:], nulons, nulats)
    m1 = m1.filled(np.nan)
    m2 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[2][:][:], nulons, nulats)
    m2 = m2.filled(np.nan)
    m3 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[3][:][:], nulons, nulats)
    m3 = m3.filled(np.nan)
    m4 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[4][:][:], nulons, nulats)
    m4 = m4.filled(np.nan)
    m5 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[5][:][:], nulons, nulats)
    m5 = m5.filled(np.nan)
    m6 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[6][:][:], nulons, nulats)
    m6 = m6.filled(np.nan)
    m7 = ResampleToNUCAPSgrid(full_lons,full_lats, totoz[7][:][:], nulons, nulats)
    m7 = m7.filled(np.nan)

    arrays = m0, m1, m2, m3, m4, m5, m6, m7
    # np.dstack altered data for some reason. Left as 8 different arrays that correspond to MERRA-2 model times. 
    # MERRA2Regridded = np.dstack(arrays)


    #Align MERRA-2 Data with NUCAPS retreval times
    Y = np.arange(0,360)
    X = np. arange(0,720)
    Z = np.arange(0,2)

    timeGridded = np.zeros((360, 720,2))
    for k in Z:
        for j in X:
            for i in Y:
                if np.isnan(nuMERRAtime[i,j,k]) == True:
                    timeGridded[i,j,k] = np.nan
                else:
                    if nuMERRAtime[i,j,k] == 0:
                        timeGridded[i,j,k] = m0[i,j]
                    if nuMERRAtime[i,j,k] == 1:
                        timeGridded[i,j,k] = m1[i,j]
                    if nuMERRAtime[i,j,k] == 2:
                        timeGridded[i,j,k] = m2[i,j]
                    if nuMERRAtime[i,j,k] == 3:
                        timeGridded[i,j,k] = m3[i,j]
                    if nuMERRAtime[i,j,k] == 4:
                        timeGridded[i,j,k] = m4[i,j]
                    if nuMERRAtime[i,j,k] == 5:
                        timeGridded[i,j,k] = m5[i,j]
                    if nuMERRAtime[i,j,k] == 6:
                        timeGridded[i,j,k] = m6[i,j]
                    if nuMERRAtime[i,j,k] == 7:
                        timeGridded[i,j,k] = m7[i,j]


    # y_var = timeGridded[:,:,:].flatten()
    y_var = timeGridded[:,:,0]
    y_var = y_var.astype(np.float32)
    # ylabel = np.append(ylabel, y_var)

    # inputlevel = np.repeat(9.2929, 259200, axis=None)
    data = np.column_stack((solar_zenith,view_angle,satellite_height,rad))
    data.astype(np.float32)
    
    imputed_target_data = impute_target(y_var)
    np.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Target/360x720_np/Total_Column_Ozone_' + str(NUCAPS_list[iteration][70:78]), imputed_target_data)
    resized_target_data = resize(imputed_target_data, (180,360,1))
    np.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Target/180x360_np/Total_Column_Ozone_' + str(NUCAPS_list[iteration][70:78]), resized_target_data)

    full_target_image = tf.keras.utils.array_to_img(imputed_target_data, data_format=None, scale=False, dtype=None)
    full_target_image.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Target/360x720_image/Total_Column_Ozone_' + str(NUCAPS_list[iteration][70:78]) + '.png',"PNG")

    small_target_image = tf.keras.utils.array_to_img(resized_target_data, data_format=None, scale=False, dtype=None)
    small_target_image.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Target/180x360_image/Total_Column_Ozone_' + str(NUCAPS_list[iteration][70:78]) + '.png',"PNG")

    data = data.reshape([360,720,133])
    imputed_input_data = impute_input(data)
    np.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Input/360x720_np/CrIS_Radiance_Input_' + str(NUCAPS_list[iteration][70:78]), imputed_input_data)
    resized_input_data = resize(imputed_input_data, (180,360,133))
    np.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Input/180x360_np/CrIS_Radiance_Input_' + str(NUCAPS_list[iteration][70:78]), resized_input_data)

    # full_input_image = tf.keras.utils.array_to_img(imputed_input_data, data_format=None, scale=False, dtype=None)
    # full_input_image.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Input/360x720_image/CrIS_Radiance_Input_' + str(NUCAPS_list[iteration][70:78]) + '.png',"PNG")

    # small_input_image = tf.keras.utils.array_to_img(resized_input_data, data_format=None, scale=False, dtype=None)
    # small_input_image.save('/mnt/nucaps-s3/philip/UNET_Model_Data/Input/180x360_image/CrIS_Radiance_Input_' + str(NUCAPS_list[iteration][70:78]) + '.png',"PNG")

    print(NUCAPS_list[iteration])
    print(MERRA_list[iteration])
