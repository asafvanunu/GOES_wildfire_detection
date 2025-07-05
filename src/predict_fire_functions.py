# %%
import rioxarray
import xarray as xr
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import joblib
from shapely.geometry import box
import matplotlib.pyplot as plt
import contextily as ctx 
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import geoviews as gv

# %%
def detect_if_CONOUS(GOES_path):
    """This function checks if the GOES file is from the CONUS region.

    Args:
        GOES_path (str): Path to the GOES file.
    """
    GOES = rioxarray.open_rasterio(GOES_path)
    scene = GOES.attrs["scene_id"]
    if scene != 'CONUS': ## Check if the scene_id is 'CONUS'
        ## If the scene_id is not 'CONUS', raise a ValueError
        raise ValueError(f"The GOES file {os.path.basename(GOES_path)} is not from the CONUS region. The scene is {scene}. Please provide a GOES file from the CONUS region.")
    

# %%
def open_MCMI(MCMI_path, band_number):
    """This function opens the MCMI NetCDF file and returns the band of the given band number

    Args:
        MCMI_path (string): The path of the MCMI NetCDF file for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc'
        band_number (int): The band number. for example 1

    Returns:
        xarray.DataArray: The band of the given band number
    """
    if (band_number < 1) or (band_number > 16): ## Check if the band number is between 1 and 16
        raise ValueError("The band number should be between 1 and 16") ## Raise an error
    
    band = f"CMI_C{band_number:02d}" ## The band name
    try:
        GOES_file = rioxarray.open_rasterio(MCMI_path) ## Open the MCMI NetCDF file
        GOES_CRS = GOES_file.rio.crs ## Get the CRS of the file
        MCMI = GOES_file.copy() ## Copy the MCMI file
    except: ## If there is an error
        print(f"Error in opening the MCMI file: {MCMI_path}") ## Print an error message
        return None ## Return None
    MCMI = MCMI.astype("float32") ## Convert the MCMI to float32
    MCMI_add_factor = MCMI[band].attrs["add_offset"] ## Get the add offset
    MCMI_scale_factor = MCMI[band].attrs["scale_factor"] ## Get the scale factor
    MCMI_fill_value = MCMI[band].attrs["_FillValue"] ## Get the fill value
    MCMI_values = MCMI[band].values[0] ## Get the values of the band
    MCMI_values[MCMI_values == MCMI_fill_value] = np.nan ## set the fill value to nan
    MCMI[band].values[0] = MCMI_values * MCMI_scale_factor + MCMI_add_factor ## Get the values of the band
    MCMI[band] = MCMI[band].rio.write_crs(GOES_CRS) ## Write the CRS of the band
    return MCMI[band] ## Return the values of the band

# %%
def open_ACM(ACM_path, product_name):
    """This function opens the ACM NetCDF file and returns the clear sky mask

    Args:
        ACM_path (string): The path of the ACM NetCDF file for example 'F:\\ML_project\\GOES_16\\ACM\\OR_ABI-L2-ACMC-M6_G16_s202301010751.nc'
        product_name (string): The product name. for example 'ACM' for 4 level classification where:
        0: Clear, 1: Probably Clear, 2: Probably Cloudy, 3: Cloudy
        and BCM for 2 level classification where:
        0: Clear, 1: Cloudy 

    Returns:
        xarray.DataArray: clear sky mask values
    """
    try:
        GOES_image = rioxarray.open_rasterio(ACM_path) ## Open the ACM NetCDF file
        GOES_CRS = GOES_image.rio.crs ## Get the CRS of the file
        ACM = GOES_image.copy() ## Copy the ACM file
        ACM = ACM.astype("float32") ## Convert the ACM to float32
        ACM_add_factor = ACM[product_name].attrs["add_offset"] ## Get the add offset
        ACM_scale_factor = ACM[product_name].attrs["scale_factor"] ## Get the scale factor
        ACM_values = ACM[product_name].values[0] * ACM_scale_factor + ACM_add_factor ## Get the values of the active fire pixels
        ACM_fill_value = ACM[product_name].attrs["_FillValue"] ## Get the fill value
        ACM_values = ACM[product_name].values[0] ## Get the values of the band
        ACM_values[ACM_values == ACM_fill_value] = np.nan ## set the fill value to nan
        ACM[product_name].values[0] = ACM_values * ACM_scale_factor + ACM_add_factor ## Get the values of the fire detection confidence
        ACM[product_name] = ACM[product_name].rio.write_crs(GOES_CRS) ## Write the CRS of the band
        return ACM[product_name] ## Return the values of the fire detection confidence
    except: ## If there is an error
        print(f"Error in opening the ACM file: {ACM_path}") ## Print an error message
        return None ## Return None

# %%
def get_GOES_date_time_from_filename(filename):
    """This function extracts the date and time from the GOES filename

    Args:
        filename (string): The GOES filename for example 'OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc'

    Returns:
        string: The date and time in the format 'YYYY-MM-DD HH:MM'
    """
    base_name = os.path.basename(filename) ## Get the base name of the file
    split_name = base_name.split(".")[0]
    split_date = split_name.split("_")[-1][1:] ## Split the name by underscore and get the last element 
    year = split_date[:4] ## Get the year
    month = split_date[4:6] ## Get the month
    day = split_date[6:8] ## Get the day
    hour = split_date[8:10] ## Get the hour
    minute = split_date[10:12] ## Get the minute
    date_time = f"{year}-{month}-{day} {hour}:{minute}" ## Create the date and time string
    return(date_time)

# %%
def fix_fill_values(cropped_GOES_image):
    """This function gets a cropped GOES image and replace the fill values with nan and return the fixed image

    Args:
        cropped_GOES_image (xr.DataArray): a rioxarray DataArray
    """
    if not isinstance(cropped_GOES_image, xr.DataArray):
        raise ValueError("cropped_GOES_image should be a rioxarray DataArray")
    else:
        GOES_fill_value = cropped_GOES_image.attrs["_FillValue"] ## Get the fill value
        cropped_GOES_image.values[0][cropped_GOES_image.values[0] == GOES_fill_value] = np.nan ## Set the fill value to nan
        return cropped_GOES_image ## Return the fixed image

# %%
def crop_GOES_using_AOI(GOES_path, GOES_band ,AOI_path):
    """This function crops the GOES file using the VIIRS file. It returns the cropped GOES file

    Args:
        GOES_path (string): The path of the GOES file. Can be MCMI, FDC, or ACM files. for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc
        GOES_band (string\int): The band of the GOES file. It can be "all" for all MCMI bands or can be 7 for MCMI. For FDC can be "Mask", "Temp", "Power". For ACM can be "Mask", "Temp", "Power". For ACM can be "ACM" or "BCM"
        AOI_path (string): The path of the AOI shapefile. for example 'F:\\ML_project\\east_us\\AOI.shp'
    """
    CMI_bands = list(range(1,17)) ## The CMI bands
    CMI_bands.append("all") ## Add all to the CMI bands
    FDC_bands = ["Mask", "Temp", "Power"] ## The FDC bands
    ACM_bands = ["ACM", "BCM"] ## The ACM bands
    band_types = CMI_bands + FDC_bands + ACM_bands ## All band types
    file_name = os.path.basename(GOES_path) ## Get the base name of the GOES file
    file_type = file_name.split("-")[2] ## Get the file type of the GOES file
    if file_type not in ["MCMIPC", "FDCC", "ACMC"]: ## If the file type is not MCMIPC, FDCC, or ACMC
        raise ValueError("The GOES file should be either MCMI, FDC, or ACM files") ## Raise an error
    if GOES_band not in band_types:
        raise ValueError("The GOES band should be either CMI for MCMI, Mask, Temp, Power for FDC, and ACM, BCM for ACM")
    
    if (file_type == "MCMIPC") and (GOES_band == "all"): ## If the file type is MCMIPC and the band is all
        GOES_image = rioxarray.open_rasterio(GOES_path) ## Open the MCMI file
        CMI_list = [f"CMI_C{band:02d}" for band in range(1, 17)] ## Get the CMI bands
    elif file_type == "MCMIPC": ## If the file type is MCMIPC
        GOES_image = open_MCMI(MCMI_path=GOES_path, band_number=GOES_band) ## Open the MCMI file
    elif file_type == "ACMC": ## If the file type is ACMC
        GOES_image = open_ACM(ACM_path=GOES_path, product_name=GOES_band) ## Open the ACM file
        
    try: ## Try to get the VIIRS polygon
        GOES_CRS = GOES_image.rio.crs ## Get the CRS of the GOES file
        AOI_polygon = gpd.read_file(AOI_path) ## Get the AOI polygon
    except: ## If there is an error
        print(f"Error in getting the AOI polygon for the AOI file: {AOI_path}") ## Print an error message
        return None
    
    try: ## Try to crop the GOES image
        AOI_polygon = AOI_polygon.to_crs(GOES_CRS) ## Convert the VIIRS polygon to the CRS of the GOES image
        GOES_cropped = GOES_image.rio.clip(AOI_polygon.geometry) ## Clip the GOES image using the VIIRS polygon
        if GOES_band == "all": ##
            GOES_cropped = GOES_cropped.astype("float32") ## Convert the MCMI to float32
            for band in CMI_list: ## For all CMI bands 
                band_add_factor = GOES_cropped[band].attrs["add_offset"] ## Get the add offset
                band_scale_factor = GOES_cropped[band].attrs["scale_factor"] ## Get the scale factor
                band_fill_value = GOES_cropped[band].attrs["_FillValue"] ## Get the fill value
                band_values = GOES_cropped[band].values[0] ## Get the values of the band
                band_values[band_values == band_fill_value] = np.nan ## set the fill value to nan
                GOES_cropped[band].values[0] = band_values * band_scale_factor + band_add_factor ## Get the values of the band
            GOES_cropped = GOES_cropped.rio.write_crs(GOES_CRS) ## Write the CRS of the band
            return GOES_cropped ## Return the cropped GOES image

        else: ## If the band is not all      
            corrected_GOES_cropped = fix_fill_values(GOES_cropped) ## Fix the fill values of the cropped GOES image
            return corrected_GOES_cropped ## Return the cropped GOES image
    except: ## If there is an error
        print(f"Error in cropping the GOES image: {GOES_path}")
        return None

# %%
def get_my_neighbores(array, row_i, col_j, distance=1, value_or_index="value"):
    """This function returns the neighbors of a pixel in a raster image

    Args:
        array (numpy array): the raster image
        row_i (int): the row index of the pixel
        col_j (int): the column index of the pixel 
        distance (int, optional): the distance of the neighbors 1 is 3x3 and 2 is 5x5. Defaults to 1.
        value_or_index (str, optional): return the value of the neighbors or the index of them. Defaults to "value".
    """
    if (0>distance) or (distance>2): ## check if the distance is between 0 and 2
        raise ValueError("The distance should be between 0 and 2")
    if value_or_index not in ["value", "index"]: ## check if the value_or_index is either value or index
        raise ValueError("The value_or_index should be either value or index")
    if isinstance(array, np.ndarray) == False: ## check if the array is a numpy array
        raise ValueError("The array should be a numpy array")
    
    array_shape = array.shape ## get the shape of the array
    if (row_i < 0) or (row_i >= array_shape[0]): ## check if the row index is within the range of the array
        raise ValueError("The row index is out of range")
    if (col_j < 0) or (col_j >= array_shape[1]): ## check if the column index is within the range of the array
        raise ValueError("The column index is out of range")
    
    pixel_loc = [row_i, col_j] ## get the location of the pixel
    if distance == 1: ## check if the distance is 1 (3x3)
        neighbors = [[row_i-1, col_j-1],
                     [row_i-1, col_j],
                     [row_i-1, col_j+1],
                     [row_i, col_j-1],
                     [row_i, col_j],
                     [row_i, col_j+1],
                     [row_i+1, col_j-1],
                     [row_i+1, col_j],
                     [row_i+1, col_j+1]]
        replace_list = [] ## create an empty list
        for i in range(len(neighbors)): ## loop through the neighbors
            ## check if the neighbors are out of the range of the array
            if (neighbors[i][0] < 0) or (neighbors[i][0] >= array_shape[0]) or (neighbors[i][1] < 0) or (neighbors[i][1] >= array_shape[1]):
                replace_list.append(neighbors[i]) ## add the neighbors to the replace_list
                
            
        if value_or_index == "value": ## check if the value_or_index is value
            list_of_values = [] ## create an empty list
            for i in range(len(neighbors)): ## loop through the neighbors
                if neighbors[i] in replace_list: ## check if the neighbors are out of the range of the array
                    list_of_values.append(np.nan) ## add nan to the list_of_values
                else: ## if the neighbors are in the range of the array
                    pixel_value = array[neighbors[i][0], neighbors[i][1]] ## get the value of the pixel
                    list_of_values.append(pixel_value) ## add the value to the list_of_values
            return list_of_values ## return the list_of_values
        elif value_or_index == "index": ## check if the value_or_index is index
            return neighbors
        
    elif distance == 2: ## check if the distance is 2 (5x5)
        neighbors = [[row_i-2, col_j-2],
                     [row_i-2, col_j-1],
                     [row_i-2, col_j],
                     [row_i-2, col_j+1],
                     [row_i-2, col_j+2],
                     [row_i-1, col_j-2],
                     [row_i-1, col_j-1],
                     [row_i-1, col_j],
                     [row_i-1, col_j+1],
                     [row_i-1, col_j+2],
                     [row_i, col_j-2],
                     [row_i, col_j-1],
                     [row_i, col_j],
                     [row_i, col_j+1],
                     [row_i, col_j+2],
                     [row_i+1, col_j-2],
                     [row_i+1, col_j-1],
                     [row_i+1, col_j],
                     [row_i+1, col_j+1],
                     [row_i+1, col_j+2],
                     [row_i+2, col_j-2],
                     [row_i+2, col_j-1],
                     [row_i+2, col_j],
                     [row_i+2, col_j+1],
                     [row_i+2, col_j+2]]
        replace_list = [] ## create an empty list
        for i in range(len(neighbors)): ## loop through the neighbors
            ## check if the neighbors are out of the range of the array
            if (neighbors[i][0] < 0) or (neighbors[i][0] >= array_shape[0]) or (neighbors[i][1] < 0) or (neighbors[i][1] >= array_shape[1]):
                replace_list.append(neighbors[i])    ## add the neighbors to the replace_list
        
            
        if value_or_index == "value": ## check if the value_or_index is value
            list_of_values = [] ## create an empty list
            for i in range(len(neighbors)): ## loop through the neighbors
                if neighbors[i] in replace_list: ## check if the neighbors are out of the range of the array
                    list_of_values.append(np.nan) ## add nan to the list_of_values
                else: ## if the neighbors are in the range of the array
                    pixel_value = array[neighbors[i][0], neighbors[i][1]] ## get the value of the pixel
                    list_of_values.append(pixel_value) ## add the value to the list_of_values
            return list_of_values ## return the list_of_values
        elif value_or_index == "index": ## check if the value_or_index is index
            return neighbors ## return the neighbors

# %%
def GOES_all_pixel_location_list(GOES_Fire_Index_array):
    """This function get GOES fire index array and return all the pixel locations in a list for example [[0,0], [0,1], ...]

    Args:
        GOES_Fire_Index_array (array): GOES Fire Index array
    """
    
    if isinstance(GOES_Fire_Index_array, np.ndarray) == False:
        raise ValueError("GOES_Fire_Index_array should be a numpy array")
    
    GOES_all_pixel_location_list = [] ## list of all the pixel locations
    image_shape = GOES_Fire_Index_array.shape ## get the shape of the GOES Fire Index array
    for i in range(image_shape[0]): ## loop through the rows
        for j in range(image_shape[1]): ## loop through the columns
            GOES_all_pixel_location_list.append([i,j]) ## append the location to the GOES_all_pixel_location_list
    return GOES_all_pixel_location_list ## return the GOES_all_pixel_location_list

# %%
def remove_cloud_neighbores(band_array, cloud_mask_array, row_i, col_j, distance,cloud_probability_list, statistic):
    """This function gets the band array, cloud mask array, row index, column index, distance, and statistic and return a statistic of the pixel neighbore withot clouds and without the pixel itself

    Args:
        band_array (array): The band array
        cloud_mask_array (array): ACM array
        row_i (int): fire pixel row index
        col_j (int): fire pixel column index
        distance (int): the buffer distance 1 for 3x3 and 2 for 5x5
        cloud_probability_list (list): list of cloud probabilities of ACM to be excluded for example [2,3]
        statistic (string): the statistic to calculate for example "mean". Avilable statistics are "mean", "median, "std, "max", "min"
    """
    if isinstance(band_array, np.ndarray) == False:
        raise ValueError("band_array should be a numpy array")
    if isinstance(cloud_mask_array, np.ndarray) == False:
        raise ValueError("cloud_mask_array should be a numpy array")

    all_clouds = -999
    all_nan = -888
    if statistic == "value": ## check if the statistic is value
        return band_array[row_i, col_j] ## return the value of the pixel
    
    else: ## if the statistic is not value
    
        band_values = get_my_neighbores(array=band_array, row_i=row_i, col_j=col_j, distance=distance, value_or_index="value") ## get the band values
        cloud_values = get_my_neighbores(array=cloud_mask_array, row_i=row_i, col_j=col_j, distance=distance, value_or_index="value") ## get the cloud values
    
        if distance == 1:
            band_values.pop(4) ## remove the center pixel
            cloud_values.pop(4) ## remove the center pixel
        elif distance == 2:
            band_values.pop(12) ## remove the center pixel
            cloud_values.pop(12) ## remove the center pixel
        
        band_values = np.array(band_values) ## convert the band values to a numpy array
        cloud_values = np.array(cloud_values) ## convert the cloud values to a numpy array
    
        is_cloud = np.isin(cloud_values, cloud_probability_list) ## check if the cloud values are in the cloud_probability_list
        filter_band_values = band_values[~is_cloud] ## filter the band values. Take only the values that are not clouds
    
        if len(filter_band_values) == 0: ## check if the filter_band_values is empty
            #print(f"in pixel {row_i}, {col_j} all of the neighbors are clouds") ## print a message
            return all_clouds ## return -999
    
        else: ## if the filter_band_values is not empty
            with warnings.catch_warnings(): ## catch the warnings
                warnings.simplefilter("ignore", category=RuntimeWarning) ## ignore the runtime warnings of nan
                if statistic == "mean": ## check if the statistic is mean
                    mean = np.nanmean(filter_band_values) ## calculate the mean of the filter_band_values
                    if np.isnan(mean): ## check if the mean is nan
                        return all_nan ## return -888
                    else: ## if the mean is not nan
                        return mean ## return the mean
                elif statistic == "median": ## check if the statistic is median
                    median = np.nanmedian(filter_band_values)
                    if np.isnan(median):
                        return all_nan
                    else:
                        return median
                elif statistic == "std": ## check if the statistic is std
                    std = np.nanstd(filter_band_values)
                    if np.isnan(std):
                        return all_nan
                    else:
                        return std
                elif statistic == "max": ## check if the statistic is max
                    max_value = np.nanmax(filter_band_values)
                    if np.isnan(max_value):
                        return all_nan
                    else:
                        return max_value
                elif statistic == "min": ## check if the statistic is min
                    min_value = np.nanmin(filter_band_values)
                    if np.isnan(min_value):
                        return all_nan
                    else:
                        return min_value

# %%
def get_fire_pixel_values_in_all_bands_for_AOI_image(pixel_location_list, MCMI_path, ACM_path, AOI_path, GOES_date_time, cloud_probability_list=[2,3]):
    """This function gets the full pixel location list and the MCMI and AOI paths and return a df with pixel values
    Args:
        pixel_location_list (list): full pixel location list for all of the image. for example [[1,2], [3,4]]
        MCMI_path (str): MCMI full path for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc'
        ACM_path (str): ACM full path for example 'F:\\ML_project\\GOES_16\\ACM\\OR_ABI-L2-ACMC-M6_G16_s202301010751.nc' 
        AOI_path (str): AOI shapefile path for example 'F:\\ML_project\\east_us\\AOI\\AOI_fire\\AOI_fire.shp'
        GOES_date_time (str): GOES date time for example '2023-01-01 07:51'
    """
    
    if isinstance(pixel_location_list, list) == False:
        raise ValueError("pixel_location_list should be a list")
    if isinstance(MCMI_path, str) == False:
        raise ValueError("MCMI_path should be a string")
    if not isinstance(ACM_path, str) and ACM_path is not None:
        raise ValueError("ACM_path should be a string or None")
    if isinstance(AOI_path, str) == False:
        raise ValueError("AOI_path should be a string")
    if isinstance(GOES_date_time, str) == False:
        raise ValueError("GOES_date_time should be a string")
    
    ## Now we will open the VIIRS file for day/night
    image_year = GOES_date_time.split("-")[0] ## get the year of the image
    image_year_int = int(image_year) ## convert the year to integer
    t = "t0" ## set the time to t0
    MCMI = crop_GOES_using_AOI(GOES_path=MCMI_path, GOES_band="all", AOI_path=AOI_path) ## crop the GOES image using the VIIRS image
    B7_values = MCMI["CMI_C07"].values[0] ## get the values of the band
    B14_values = MCMI["CMI_C14"].values[0] ## get the values of the band
    FI = (B7_values - B14_values) / (B7_values + B14_values) ## calculate the fire index
    if ACM_path != None: ## check if the ACM path is not None
        if image_year_int <= 2021: ## check if the image year is less than or equal to 2021
            cloud_probability_list = [1] ## set the cloud probability list to [1] for 2021 or earlier
            ACM = crop_GOES_using_AOI(GOES_path=ACM_path, GOES_band="BCM", AOI_path=AOI_path) ## crop the GOES image using the AOI
            ## The cropping is on the BCM band
        else: ## if the image year is not 2021 open ACM
            ACM = crop_GOES_using_AOI(GOES_path=ACM_path, GOES_band="ACM", AOI_path=AOI_path) ## crop the GOES image using the AOI
            ## the cropping is on the ACM band
        ACM_values = ACM.values[0] ## get the values of the ACM
    else: ## if the ACM path is None, that means the user does not have ACM data
        ACM_values = np.zeros_like(FI) ## create an array of zeros with the same shape as the FI array
    band_list = [f"CMI_C{band:02d}" for band in range(1, 17)] ## Get the CMI bands
    indices_list = ["FI"] ## list of indices for example fire index (FI)
    band_iteration_list = band_list + indices_list ## combine the band_list and indices_list
    statistics_list = ["value", "mean", "median", "std","min","max"] ## list of statistics for example value, mean, median, std
    
    row_list = [] ## list to store the row values
    col_list = [] ## list to store the column values
    ACM_list = [] ## list to store the ACM values
    
    for loc in pixel_location_list: ## loop through the pixel location list
        row = loc[0] ## get the row location
        col = loc[1] ## get the column location
        row_list.append(row) ## append the row to the row_list
        col_list.append(col) ## append the column to the col_list
        ACM_value = ACM_values[row, col] ## get the value of the ACM
        ACM_list.append(ACM_value) ## append the ACM value to the ACM_list
        
    d = {} ## dictionary to store the values
    d["row"] = row_list ## add the row_list to the dictionary
    d["col"] = col_list ## add the col_list to the dictionary
    for band in band_iteration_list: ## loop through the band_iteration_list
        if band == "FI": ## check if the band is FI
            FI_value_list = [] ## list to store the fire index values
            FI_n_mean_list = [] ## list to store the fire index mean values
            FI_n_median_list = [] ## list to store the fire index median values
            FI_n_std_list = [] ## list to store the fire index std values
            FI_n_min_list = [] ## list to store the fire index min values
            FI_n_max_list = [] ## list to store the fire index max values
            for loc in pixel_location_list: ## loop through the pixel location list
                row = loc[0] ## get the row location
                col = loc[1] ## get the column location
                for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                    stat_value = remove_cloud_neighbores(band_array=FI, 
                                                              cloud_mask_array=ACM_values,
                                                              row_i=row,
                                                              col_j=col,
                                                              distance=1,
                                                              cloud_probability_list=cloud_probability_list,
                                                              statistic=stat) ## get the neighbors of the pixel
                    if stat == "value":
                        FI_value_list.append(stat_value) ## append the value to the FI_value_list
                    elif stat == "mean": ## check if the stat is mean
                        FI_n_mean_list.append(stat_value) ## append the mean to the FI_n_mean_list
                    elif stat == "median": ## check if the stat is median
                        FI_n_median_list.append(stat_value) ## append the median to the FI_n_median_list
                    elif stat == "std": ## check if the stat is std
                        FI_n_std_list.append(stat_value) ## append the std to the FI_n_std_list
                    elif stat == "min": ## check if the stat is min
                        FI_n_min_list.append(stat_value) ## append the min to the FI_n_min_list
                    elif stat == "max": ## check if the stat is max
                        FI_n_max_list.append(stat_value) ## append the max to the FI_n_max_list
            d[f"{t}_FI_value"] = FI_value_list ## add the FI_value_list to the dictionary
            d[f"{t}_FI_mean"] = FI_n_mean_list ## add the FI_n_mean_list to the dictionary
            d[f"{t}_FI_median"] = FI_n_median_list ## add the FI_n_median_list to the dictionary
            d[f"{t}_FI_std"] = FI_n_std_list ## add the FI_n_std_list to the dictionary
            d[f"{t}_FI_min"] = FI_n_min_list ## add the FI_n_min_list to the dictionary
            d[f"{t}_FI_max"] = FI_n_max_list ## add the FI_n_max_list to the dictionary
        else: ## if the band is not FI
            band_number = f'B{band.split("_")[-1][1:]}' ## get the band number for example B01
            ## crop the GOES image using the VIIRS image
            B = MCMI[band] ## get the band
            band_array = B.values[0] ## get the values of the band
            band_value_list = [] ## list to store the band values
            band_n_mean_list = [] ## list to store the band mean values
            band_n_median_list = [] ## list to store the band median values
            band_n_std_list = [] ## list to store the band std values
            band_n_min_list = [] ## list to store the band min values
            band_n_max_list = [] ## list to store the band max values
            for loc in pixel_location_list: ## loop through the pixel location list
                row = loc[0] ## get the row location
                col = loc[1] ## get the column location
                for stat in statistics_list: ## loop through the statistics_list
                    ## get the neighbors of the pixel including the pixel itself
                    stat_value = remove_cloud_neighbores(band_array=band_array, 
                                                              cloud_mask_array=ACM_values,
                                                              row_i=row,
                                                              col_j=col,
                                                              distance=1,
                                                              cloud_probability_list=cloud_probability_list,
                                                              statistic=stat) ## get the neighbors of the pixel
                    if stat == "value": ## check if the stat is value
                        band_value_list.append(stat_value) ## append the value to the band_value_list
                    elif stat == "mean": ## check if the stat is mean
                        band_n_mean_list.append(stat_value) ## append the mean to the band_n_mean_list
                    elif stat == "median": ## check if the stat is median
                        band_n_median_list.append(stat_value) ## append the median to the band_n_median_list
                    elif stat == "std": ## check if the stat is std
                        band_n_std_list.append(stat_value) ## append the std to the band_n_std_list
                    elif stat == "min": ## check if the stat is min
                        band_n_min_list.append(stat_value)
                    elif stat == "max": ## check if the stat is max
                        band_n_max_list.append(stat_value)
            d[f"{t}_{band_number}_value"] = band_value_list ## add the band_value_list to the dictionary
            d[f"{t}_{band_number}_mean"] = band_n_mean_list ## add the band_n_mean_list to the dictionary
            d[f"{t}_{band_number}_median"] = band_n_median_list ## add the band_n_median_list to the dictionary
            d[f"{t}_{band_number}_std"] = band_n_std_list ## add the band_n_std_list to the dictionary
            d[f"{t}_{band_number}_min"] = band_n_min_list ## add the band_n_min_list to the dictionary
            d[f"{t}_{band_number}_max"] = band_n_max_list ## add the band_n_max_list to the dictionary
        
    df = pd.DataFrame(d) ## create a DataFrame from the dictionary
    df[f"{t}_ACM_value"] = ACM_list ## add the ACM_list to the DataFrame
    file_name = os.path.basename(MCMI_path).split("_")[-1] ## get the base name of the MCMI file 
    file_name_list = np.repeat(file_name, len(df)) ## repeat the file name for the length of the DataFrame
    date_time_list = np.repeat(GOES_date_time, len(df)) ## repeat the date time for the length of the DataFrame
    df.insert(0, f"{t}_MCMI_file", file_name_list) ## insert the file name to the first column
    df.insert(1, f"{t}_GOES_date_time", date_time_list) ## insert the date time to the second column
    return df ## return the DataFrame

# %%
def create_AOI_image_df(MCMI_path,
                        AOI_path, 
                          ACM_path=None
                          ):
    """This function gets the MCMI, ACM, AOI paths return the AOI image DataFrame

    Args:
        MCMI_path (str): path to the MCMI file for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIPC-M6_G16_s202301010751.nc'
        AOI_path (str): path of the AOI shapefile for example 'F:\\ML_project\\east_us\\AOI\\AOI_fire\\AOI_fire.shp'
        ACM_path (str): path of the ACM file for example 'F:\\ML_project\\GOES_16\\ACM\\OR_ABI-L2-ACMC-M6_G16_s202301010751.nc'
    """
    ## check the input types
    if isinstance(MCMI_path, str) == False:
        raise ValueError("MCMI_path should be a string")
    if not isinstance(ACM_path, str) and ACM_path is not None:
        raise ValueError("ACM_path should be a string or None")
    if isinstance(AOI_path, str) == False:
        raise ValueError("AOI_path should be a string")

    cloud_probability_list=[2,3] ## set the cloud probability list to [2,3] by default
    GOES_date_time = get_GOES_date_time_from_filename(MCMI_path) ## get the GOES date time from the MCMI file name
    ## Open GOES bands
    MCMI = crop_GOES_using_AOI(GOES_path=MCMI_path, GOES_band="all", AOI_path=AOI_path) ## crop the GOES image using the VIIRS image
    B7 = MCMI["CMI_C07"] ## get the band 7
    B14 = MCMI["CMI_C14"] ## get the band 14
    FI = (B7.values[0] - B14.values[0])/(B7.values[0] + B14.values[0]) ## calculate the fire index

    
    GOES_pixel_list = GOES_all_pixel_location_list(GOES_Fire_Index_array=FI)
    
    ## Get the pixel values

    df_pixels = get_fire_pixel_values_in_all_bands_for_AOI_image(pixel_location_list=GOES_pixel_list,
                                        MCMI_path=MCMI_path,
                                        ACM_path=ACM_path,
                                        AOI_path=AOI_path,
                                        GOES_date_time=GOES_date_time,
                                        cloud_probability_list=cloud_probability_list)
    
    return df_pixels ## return the train_df


# %%
def open_catboost_model(catboost_path):
    """This function opens a catboost model from a given path

    Args:
        catboost_path (str): The path to the catboost model file
    """
    if isinstance(catboost_path, str) == False:
        raise ValueError("catboost_path should be a string")
    
    grid_search = joblib.load(catboost_path) ## Load the grid search object
    ML_model = grid_search.best_estimator_ ## Get the best estimator from the grid search
    return ML_model ## Return the ML model
    

# %%
def filter_df_for_ML_model(df):
    """This function filters the DataFrame for the ML model. It removes the columns that are not needed for the ML model

    Args:
        df (pd.DataFrame): The DataFrame to filter
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a pandas DataFrame")
    
    name_filter_list = [] ## create a list to keep the names of the columns
    for band in ["01","02", "03","04","05","06","07","08","09","10","11","12", "13","14","15","16","FI"]:
        for stat in ["value", "mean", "median", "std", "max", "min"]: ## loop over statistics
            if band == "FI": ## if the band is FI, add the name to the list
                name_filter_list.append(f"t0_FI_{stat}") ## add the name to the list
            else: ## if the band is not FI, add the name to the list
                name_filter_list.append(f"t0_B{band}_{stat}") ## add the name to the list
                
    filtered_df = df[name_filter_list] ## filter the DataFrame using the name_filter_list
    return filtered_df ## return the filtered DataFrame

# %%
def make_raster_template(cropped_MCMI):
    """This function creates a raster template from the cropped MCMI image. It returns the raster template

    Args:
        cropped_MCMI (xarray.DataArray): The cropped MCMI image
    """
    if not isinstance(cropped_MCMI, xr.DataArray):
        raise ValueError("cropped_MCMI should be an xarray DataArray")
    
    raster_template = cropped_MCMI.copy() ## Create a copy of the cropped ACM image
    raster_template.name = "Fire prediction" ## Set the name of the raster template
    raster_template.attrs["flag_meanings"] = ["No Fire, Fire"] ## Set the flag meanings of the ACM image
    raster_template.attrs["flag_values"] = [0, 1] ## Set the flag values of the image
    raster_template.attrs["add_offset"] = 0
    raster_template.attrs["scale_factor"] = 1
    raster_template.attrs.pop('long_name') ## Remove the long name attribute from the raster template
    raster_template.attrs.pop('ancillary_variables') ## Remove the ancillary variables attribute from the raster template
    raster_template.attrs.pop('cell_methods')
    raster_template.attrs.pop('coordinates')
    raster_template.attrs.pop('resolution')
    raster_template.attrs.pop('sensor_band_bit_depth')
    raster_template.attrs.pop('standard_name') 
    raster_template.attrs.pop('units')
    raster_template.attrs.pop('valid_range')
    raster_template.attrs.pop('_Unsigned')    
    raster_template.attrs["_FillValue"] = np.nan ## Set the fill value of the raster template to nan
    raster_template = raster_template.squeeze() ## Squeeze the raster template to remove the singleton dimension
    return raster_template ## Return the raster template


# %%
def GOES_extent_poly(MCMI_path):
    """This function gets the MCMI path and returns the extent polygon of the MCMI image

    Args:
        ACM_path (str): The path to the ACM image file for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-MCMIC-M6_G16_s202301010751.nc'
    """
    if not isinstance(MCMI_path, str):
        raise ValueError("MCMI_path should be a string")
    
    MCMI = rioxarray.open_rasterio(MCMI_path) ## Open the ACM image
    extent_geom = box(*MCMI.rio.bounds())
    gdf = gpd.GeoDataFrame({"geometry": [extent_geom]}, crs=MCMI.rio.crs) ## Create a GeoDataFrame with the extent geometry
    return gdf ## Return the GeoDataFrame with the extent polygon
    

# %%
def AOI_inside_GOES_extent(AOI_path, MCMI_path):
    """This function checks if the AOI is inside the GOES extent polygon. returns True if the AOI is inside the GOES extent polygon, otherwise returns False

    Args:
        AOI_path (str): The path to the AOI shapefile for example 'F:\\ML_project\\east_us\\AOI\\AOI_fire\\AOI_fire.shp'
        MCMI_path (str): The path to the ACM image file for example 'F:\\ML_project\\GOES_16\\MCMI\\OR_ABI-L2-ACMC-M6_G16_s202301010751.nc'
    """
    if not isinstance(AOI_path, str):
        raise ValueError("AOI_path should be a string")
    if not isinstance(MCMI_path, str):
        raise ValueError("MCMI_path should be a string")
    ## Read the AOI shapefile and get the GOES extent polygon
    GOES_poly = GOES_extent_poly(MCMI_path) ## Get the GOES extent polygon
    AOI_poly = gpd.read_file(AOI_path) ## Read the AOI shap
    AOI_poly = AOI_poly.to_crs(GOES_poly.crs) ## Convert the AOI polygon to the CRS of the GOES extent polygon
    
    if len(AOI_poly)>1: ## If the AOI has more than one polygon
        result_series = AOI_poly.geometry.apply(lambda x: GOES_poly.contains(x).any())
        inside = np.all(result_series) ## Check if all the AOI polygons are inside the GOES extent polygon
        return inside ## Return True if all the AOI polygons are inside the GOES extent polygon, otherwise return False
    else: ## If the AOI has only one polygon
        inside = np.all(GOES_poly.contains(AOI_poly.geometry)) ## Check if the AOI polygon is inside the GOES extent polygon
        return inside ## Return True if the AOI polygon is inside the GOES extent polygon, otherwise return False

# %%
def predict_fire_for_AOI(MCMI_path:str,
                         AOI_path:str,
                         ACM_path:str = None,
                         save_raster:bool = True,
                         output_path:str = None):
    """This function predicts fire for a given AOI using the GOES MCMI and ACM images.
    The image must be in CONOUS mode and the AOI must be inside the GOES extent polygon. The AOI must be a shapefile with '.shp' extension.
    It returns the prediction raster and saves it to the output path if save_raster is True.

    Args:
        MCMI_path (str): Path to the MCMI file for example 'data\\GOES_18\\OR_ABI-L2-MCMIPC-M6_G18_s202407071036.nc'
        ACM_path (str): Path to the ACM file for example 'data\\GOES_18\\OR_ABI-L2-ACMC-M6_G18_s202407071036.nc' if there is no ACM file, set it to None
        AOI_path (str, optional): Path to AOI shapefile for example 'data\\GOES_18\\AOI\\AOI.shp'.
        save_raster (bool, optional): An option to save the prediction as a raster . Defaults to True.
        output_path (str, optional): The output path of the raster. Should be an existing path. Defaults to None.
    """
    if not isinstance(MCMI_path, str):
        raise ValueError("MCMI_path should be a string")
    if not isinstance(ACM_path, str) and ACM_path is not None:
        raise ValueError("ACM_path should be a string or None")
    if AOI_path is not None and not isinstance(AOI_path, str):
        raise ValueError("AOI_path should be a string if provided")
    if AOI_path.endswith(".shp") == False:
        raise ValueError("AOI_path should be a shapefile path ending with .shp")
    if not isinstance(save_raster, bool):
        raise ValueError("save_raster should be a boolean")
    if output_path is not None and not isinstance(output_path, str):
        raise ValueError("output_path should be a string if provided")
    if save_raster == True and output_path is None:
        raise ValueError("If save_raster is True, output_path should be provided")
    
    detect_if_CONOUS(MCMI_path) ## Check if the MCMI image is in CONOUS mode
    MCMI_time = get_GOES_date_time_from_filename(MCMI_path) ## Get the GOES date time from the MCMI file name
    if ACM_path is not None:
        detect_if_CONOUS(ACM_path) ## Check if the ACM image is in CONOUS mode
        ACM_time = get_GOES_date_time_from_filename(ACM_path) ## Get the GOES date time from the ACM file name
        if MCMI_time != ACM_time: ## Check if the GOES date times are
            raise ValueError(f"MCMI and ACM images should have the same GOES date time. MCMI time: {MCMI_time}, ACM time: {ACM_time}") ## Raise an error if the GOES date times are not the same
        
    if not AOI_inside_GOES_extent(AOI_path=AOI_path, MCMI_path=MCMI_path): ## Check if the AOI is inside the GOES extent polygon
        raise ValueError("AOI is not inside the GOES extent. Please provide a valid AOI")
    
    print(f"Now working of GOES time stamp: {MCMI_time}") ## Print the GOES date time
    
    try:
        AOI_df = create_AOI_image_df(MCMI_path=MCMI_path, ACM_path=ACM_path, AOI_path=AOI_path) ## Create the AOI image DataFrame
        AOI_df = filter_df_for_ML_model(AOI_df) ## Filter the DataFrame for the ML model
        print(f"data successfully created for GOES time stamp: {MCMI_time}") ## Print a message if the DataFrame is created successfully
    except Exception as e: ## Catch any exception that occurs during the creation of the AOI image DataFrame
        raise ValueError(f"Error creating AOI image DataFrame: {e}") ## Raise an error with the exception message
    
    try:
        MCMI = crop_GOES_using_AOI(GOES_path=MCMI_path, GOES_band=7, AOI_path=AOI_path) ## Crop the MCMI image using the AOI
    except Exception as e: ## Catch any exception that occurs during the cropping of the MCMI image
        raise ValueError(f"Error cropping MCMI image: {e}")
    
    catboost_path = r"model\catboost_model.pkl"
    ML_model = open_catboost_model(catboost_path) ## Open the catboost model
    threshold = 0.9 ## Set the threshold for the prediction
    y_prob = ML_model.predict_proba(AOI_df)[:,1] ## predict the probabilities of fire
    y_pred = (y_prob >= threshold).astype(int) ## predict the fire labels using the threshold
    prediction_raster = make_raster_template(cropped_MCMI=MCMI) ## Create the raster template from the cropped ACM image
    array_shape = prediction_raster.shape ## Get the shape of the raster template
    pred_array = y_pred.reshape(array_shape) ## Reshape the predicted labels to the shape of the raster template
    prediction_raster.values = pred_array ## Set the predicted labels to the prediction image
    prediction_raster = prediction_raster.astype("float32") ## Convert the raster image to float32
    prediction_raster.values[np.isnan(MCMI.values[0])] = np.nan ## Set the values of the raster image to nan where the ACM image is nan
    print(f"Prediction raster created for GOES time stamp: {MCMI_time}") ## Print a message if the prediction raster is created successfully
    if save_raster == True: ## Check if the rsave_raster is True
        prediction_raster.rio.to_raster(f"{output_path}.tif", driver="GTiff") ## Save the raster image to the output path
        print(f"Prediction raster saved to {output_path}.tif") ## Print a message if the raster image is saved successfully
        return prediction_raster ## Return the prediction raster
    elif save_raster == False: ## If save_raster is False
        print("save_raster is False, returning the prediction raster without saving")
        return prediction_raster

# %%
def create_polygon_from_latlon(x_min, x_max, y_min, y_max, output_path):
    """
    Creates a GeoDataFrame with a single polygon based on latitude and longitude bounds and save it in an exsiting path.

    Args:
        x_min (float): Minimum longitude (west)
        x_max (float): Maximum longitude (east)
        y_min (float): Minimum latitude (south)
        y_max (float): Maximum latitude (north)
        output_path (str): Path to save the GeoDataFrame as a shapefile.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the bounding box polygon.
    """
    crs = 'EPSG:4326'  # Default CRS for latitude/longitude
    polygon = box(x_min, y_min, x_max, y_max) ## Create a polygon using the bounding box coordinates
    gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs) ## Create a GeoDataFrame with the polygon geometry
    gdf.to_file(f"{output_path}.shp")  ## Save the GeoDataFrame as a shapefile
    print(f"Polygon saved to {output_path}.shp")  ## Print a message


# %%
AOI_path = r"data\\AOI\\AOI.shp" ## Path to the AOI shapefile
MCMI_path = r"data\\GOES_18\\OR_ABI-L2-MCMIPC-M6_G18_s202407071036.nc" ## Path to the MCMI file

# %%
x = predict_fire_for_AOI(MCMI_path=MCMI_path,
                            AOI_path=AOI_path,
                            ACM_path=None,
                            save_raster=False)
                         
                         

# %%
def plot_fire_prediction(fire_prediction_raster, AOI):
    """This function plots the fire prediction raster and the AOI polygon on a map.

    Args:
        fire_prediction_raster (xarray.DataArray): The fire prediction raster.
        AOI (geopandas.GeoDataFrame): The AOI polygon.
    """
    if not isinstance(fire_prediction_raster, xr.DataArray):
        raise ValueError("fire_prediction_raster should be an xarray DataArray")
    if not isinstance(AOI, gpd.GeoDataFrame):
        raise ValueError("AOI should be a geopandas GeoDataFrame")
    


    # Reproject to EPSG:3857 if needed
    fire_prediction_raster = fire_prediction_raster.rio.reproject("EPSG:3857")
    AOI = AOI.to_crs(epsg=3857)

    # Define binary colormap: 0 = transparent black, 1 = solid red
    binary_cmap = ListedColormap([
        (0, 0, 0, 0.6),   # No Fire (transparent black)
        (1, 0, 0, 1.0)    # Fire (solid red)
    ])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    fire_prediction_raster.plot(ax=ax, cmap=binary_cmap, vmin=0, vmax=1, add_colorbar=False)

    AOI.boundary.plot(ax=ax, edgecolor='purple', linewidth=4)

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, alpha=0.5)

    # Remove axis ticks, labels, titles
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    # Legend (colors won't show transparency in legend)
    legend_elements = [
        mpatches.Patch(color='black', label='No Fire (0)'),
        mpatches.Patch(color='red', label='Fire (1)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=False)

    plt.tight_layout()
    plt.show()

# %%
def plot_fire_prediction_on_interactive_map(prediction_raster, AOI):
    """This function plots the fire prediction raster on an interactive map.

    Args:
        prediction_raster (xarray.DataArray): The fire prediction raster.
        AOI (geopandas.GeoDataFrame): The AOI polygon.
    """
    if not isinstance(prediction_raster, xr.DataArray):
        raise ValueError("prediction_raster should be an xarray DataArray")
    if not isinstance(AOI, gpd.GeoDataFrame):
        raise ValueError("AOI should be a geopandas GeoDataFrame")
    
    # Optional: Reproject raster for basemap compatibility
    prediction_raster = prediction_raster.rio.reproject("EPSG:3857")
    AOI = AOI.to_crs(epsg=3857)
    prediction_raster.coords["x"].attrs["units"] = "Longitude"  
    prediction_raster.coords["y"].attrs["units"] = "Latitude"
    
    gv.extension('bokeh')


    # Define a simple binary colormap: 0 = black, 1 = red

    binary_cmap = ['black', 'red']

    vector = gv.Polygons(AOI, crs=gv.tile_sources.EsriImagery.crs).opts(
        fill_alpha=0,
        line_color='purple',
        line_width=4,
    )
    
    raster = gv.Image(prediction_raster, crs=gv.tile_sources.EsriImagery.crs).opts(
        cmap=binary_cmap,
        colorbar=True,
        color_levels=2,
        clim=(0, 1),
        tools=['hover'],
        alpha=0.6,  # transparency so basemap is visible
        colorbar_opts={'title': 'Fire Prediction: 0 = No Fire, 1 = Fire'}, 
    )

    plot = raster * vector * gv.tile_sources.EsriImagery
    plot.opts(width=800, height=600, title="Fire Prediction Map")
    return plot


