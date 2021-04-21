'''
This library is used to preprocess raw images (resizing, denoising). 
This library also stores statistical parameters of images for automated image quality filtering.
The input for the library is relative path for raw image folder and resized image folder.
'''
#import all libraries
import os, glob
import pandas as pd
import numpy as np
import cv2, csv, json, math, time, argparse
from pathlib import Path
from PIL import Image, ImageOps,ImageFilter
from scipy import ndimage  
import imageio 
from pylab import *

# Pandas column names for storing resized and preprocessed images
logo_folder_columns = ['true_logo','folder_name','folder_img_count']
# Pandas column names for storing statistical parameters for image quality filtering
resized_image_columns = ['true_logo','logo_img_name','original_image_name','company_brand','mean_mean_val','mean_std_val','mscn_val']
# Sampling images for balancing the images from each class
SAMPLE_SIZE = 200
EXPECTED_DIMENSION=120
SIZE_THRESHOLD=17

#Get folder information and logo brand names
def Get_folder_name_labels(dir_path):

    df_logos_folder = pd.DataFrame(columns=logo_folder_columns)
    mn = 0
    folders = ([name for name in sorted(os.listdir(dir_path), key=str.casefold)]) # get all directories 
    num_folders =1

    for company_name in folders:
        contents = os.listdir(os.path.join(dir_path,company_name)) # get list of contents
        if len(contents) <= mn: # if greater than the limit, print folder and number of contents
            continue
        row = pd.Series({logo_folder_columns[0] :num_folders,
                         logo_folder_columns[1] :company_name,
                        logo_folder_columns[2]: len(contents),
                        'folder_path': os.path.join(dir_path,company_name),})
        print(os.path.join(dir_path,company_name))
        print("length:",str(len(contents)))
        num_folders = num_folders+1
        df_logos_folder = df_logos_folder.append(row,ignore_index=True)   
        
    
    return df_logos_folder

#Image processing for resizing images
def fix_aspect_ratio(img):
    original_max_dim = max(float(img.size[0]),float(img.size[1]))
    original_min_dim = min(float(img.size[0]),float(img.size[1]))
    wpercent = (EXPECTED_DIMENSION/float(original_max_dim))
    hsize = int((original_min_dim*float(wpercent)))
    new_im = img.resize((EXPECTED_DIMENSION,hsize), Image.ANTIALIAS)
    return new_im


# Make square images
def make_square(img, min_size=120, fill_color=(0, 0, 0, 0)):
    x, y = img.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

# Sharpen the edges
def sharpen_filter(img):
    sharp_im = img.filter(ImageFilter.SHARPEN)
    return sharp_im

# Statistical parameters for images
def calculate_MSCN_val(img):
    C=3.0/255.0
    blurred_img=cv2.GaussianBlur(img, (7, 7), 1.166)
    blurred_sq = blurred_img * blurred_img 
    sigma = cv2.GaussianBlur(img * img, (7, 7), 1.166) 
    sigma = (sigma - blurred_sq) ** 0.5
    sigma = sigma + C
    MCSN_value = (img - blurred_img)/sigma
    return MCSN_value

# Load dataset of logos in a dataframe
def resize_images_df(folder_path_global, company_name,dir_litw_resized,folder_logo_num):
    df_logos_pickle = pd.DataFrame(columns=resized_image_columns)
    
     # in each folder, find image file and resize-scale them without distortion
    index_image= 1;
    glob_path_complete_path_inter=os.path.join(folder_path_global,company_name)
    glob_path_complete_path=Path(glob_path_complete_path_inter)
    mod_folder_name = 'Mod-' + str(company_name)
    mod_full_dir = os.path.join(dir_litw_resized,mod_folder_name) 
    brand_count_row=1
    for filename_logo in glob_path_complete_path.glob('**/*.jpg'):
         with open(filename_logo) as imagefile:
            # Choose the classes with enough samples
            if brand_count_row > SAMPLE_SIZE:
                return df_logos_pickle
            image_original = Image.open(filename_logo)
            dir_name,file_orig_name = os.path.split(filename_logo)
            get_file_data_shape_x, get_file_data_shape_y = image_original.size
            max_size_dimension = max(float(get_file_data_shape_x),float(get_file_data_shape_y))
            if max_size_dimension < SIZE_THRESHOLD: # size should be above the threshold
                continue 
            img_mod_name = company_name +'-AB-'+ str(index_image) +'.png'
            
            os.makedirs(mod_full_dir, exist_ok=True)
            path_to_image = os.path.join(mod_full_dir,img_mod_name)             
            
            company_brand_name = company_name         
           
            # Image preprocessing
            image_aspect=fix_aspect_ratio(image_original)          
            img_new = make_square(image_aspect)       
            sharpen_image=sharpen_filter(img_new)
            im_med = ndimage.median_filter(sharpen_image, 3)
            
            # Statistical information
            norm_image=cv2.normalize(im_med,None,0,1, cv2.NORM_MINMAX)          
            mean_values, std_values = cv2.meanStdDev(norm_image)
            mscn_values= calculate_MSCN_val(norm_image)
            mean_mean_val = np.mean(mean_values)
            mean_std_val = np.mean(std_values)
            mean_mscn_val =  np.mean(mscn_values)
            company_brand_name = company_name
            
            row = pd.Series({resized_image_columns[0] :folder_logo_num,
                             resized_image_columns[1] :img_mod_name,
                             resized_image_columns[2] :file_orig_name,
                             resized_image_columns[3] :company_brand_name,
                            resized_image_columns[4]: mean_mean_val,
                            resized_image_columns[5]: mean_std_val,
                            resized_image_columns[6]: mean_mscn_val,})
            
            # save information and save resized images
            df_logos_pickle = df_logos_pickle.append(row,ignore_index=True)           
            brand_count_row=brand_count_row+1
            file_name=os.path.join(mod_full_dir,img_mod_name)
            imageio.imwrite(file_name, im_med)
            index_image =index_image+1
            
  

    return df_logos_pickle

# Access all the logo folders and process the image files
def get_all_images(dir_litw,dir_litw_resized):
    df_list=[]
    df_folder_Stat_list=[]
    folder_logo_num=0
    folders = ([name for name in sorted(os.listdir(dir_litw), key=str.casefold)])  
    for company_name in folders:
        folder_path_global=dir_litw
        df_list.append(resize_images_df(folder_path_global, company_name,dir_litw_resized, folder_logo_num))
        tp_df = pd.concat(df_list, ignore_index=True)  
        folder_logo_num=folder_logo_num+1
    return tp_df


