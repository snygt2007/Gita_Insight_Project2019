'''
This library is used to preprocess raw images (resizing, denoising) for semi-supervised learning. 
The input for the library is relative path for raw image folder and resized image folder.
Ref : MSCN values are calculated based on https://www.learnopencv.com/image-quality-assessment-brisque/
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
logo_folder_columns = ['company_brand','folder_mean_val','folder_std_val','folder_mscn_val']
# Pandas column names for storing statistical parameters for image quality filtering
processed_image_columns =['true_logo','logo_img_name','original_image_name','company_brand','mean_mean_val','mean_std_val','mscn_val']
feature_storage_columns =['true_logo', 'logo_img_name', 'company_brand', 'features_list_count']
# Sampling images for balancing the images from each class
SAMPLE_SIZE = 200
EXPECTED_DIMENSION=120
SIZE_THRESHOLD=17

#Image processing for resizing images
def fix_aspect_ratio(img):
    original_max_dim = max(float(img.size[0]),float(img.size[1]))
    original_min_dim = min(float(img.size[0]),float(img.size[1]))
    wpercent = (EXPECTED_DIMENSION/float(original_max_dim))
    hsize = int((original_min_dim*float(wpercent)))
    new_im = img.resize((EXPECTED_DIMENSION,hsize), Image.ANTIALIAS)
    return new_im

# Make square images
def make_square(img, min_size=EXPECTED_DIMENSION, fill_color=(0, 0, 0, 0)):
    x, y = img.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

# Sharpen the edges
def sharpen_filter(img):
    sharp_im = img.filter(ImageFilter.SHARPEN)
    return sharp_im

# Statistical values
def calculate_MSCN_val(img):
    C=3.0/255.0
    blurred_img=cv2.GaussianBlur(img, (7, 7), 1.166)
    blurred_sq = blurred_img * blurred_img 
    sigma = cv2.GaussianBlur(img * img, (7, 7), 1.166) 
    sigma = (sigma - blurred_sq) ** 0.5
    sigma = sigma + C
    MCSN_value = (img - blurred_img)/sigma
    return MCSN_value

# Get folder statistics
def get_folder_stats(df_logos_folder_label, df_folder_details,dir_litw_resized):
    
    
    df_logos_folder = pd.DataFrame(columns=logo_folder_columns) 
    
    folders = ([name for name in sorted(os.listdir(dir_litw_resized), key=str.casefold)]) # get all directories 
    
    for company_name in folders:
        company_name=company_name[4:len(company_name)]
        
        df_rows=df_folder_details.loc[df_folder_details['company_brand'] == company_name]
        
        mean_mean_val=df_rows["mean_mean_val"].mean() 
        mean_std_val=df_rows["mean_std_val"].mean()
        mean_mscn_val=df_rows["mscn_val"].mean() 
        row = pd.Series({logo_folder_columns[0] :company_name,
                            logo_folder_columns[1]: mean_mean_val,
                            logo_folder_columns[2]: mean_std_val,
                            logo_folder_columns[3]: mean_mscn_val,})
            
        df_logos_folder = df_logos_folder.append(row,ignore_index=True) 
    return df_logos_folder

# Load dataset of logos in a dataframe
# Semisupervised image quality threshold is different than the supervised image quality requirements
def get_file_excluded_1(folder_path_global, company_name,dir_litw_resized, folder_logo_num, threshold_mean,threshold_std,threshold_mscn):
    df_logos_pickle = pd.DataFrame(columns=processed_image_columns)
    
     # in each folder, find image file and resize-scale them without distortion
    index_image= 1;
    #print(glob_path_complete) 
    glob_path_complete_path_inter=os.path.join(folder_path_global,company_name)
    glob_path_complete_path=Path(glob_path_complete_path_inter)
    mod_folder_name = str(company_name)
    mod_full_dir = os.path.join(dir_litw_resized,mod_folder_name) 
    brand_count_row=1
    
    for filename_logo in glob_path_complete_path.glob('**/*.png'):
         with open(filename_logo) as imagefile:
          
            image_original = Image.open(filename_logo)
            dir_name,file_orig_name = os.path.split(filename_logo)
            get_file_data_shape_x, get_file_data_shape_y = image_original.size
            max_size_dimension = max(float(get_file_data_shape_x),float(get_file_data_shape_y))
            if max_size_dimension < SIZE_THRESHOLD:
                continue 
            
            img_mod_name = company_name +'-AB-'+ str(index_image) +'.png'
          
            company_brand_name = company_name         
            # Image processing
            image_aspect=fix_aspect_ratio(image_original)          
            img_new = make_square(image_aspect)            
            sharpen_image=sharpen_filter(img_new)
            im_med = ndimage.median_filter(sharpen_image, 3)
            norm_image=cv2.normalize(im_med,None,0,1, cv2.NORM_MINMAX)
            
            mean_values, std_values = cv2.meanStdDev(norm_image)
            mscn_values= calculate_MSCN_val(norm_image)
            mean_mean_val = np.mean(mean_values)
            mean_std_val = np.mean(std_values)
            mean_mscn_val =  np.mean(mscn_values)
            company_brand_name = company_name
            # Filter images based on threshold
            if mean_mean_val < threshold_mean or mean_std_val < threshold_std or abs(mean_mscn_val) < abs(threshold_mscn):
                continue
                   
            os.makedirs(mod_full_dir, exist_ok=True)
            path_to_image = os.path.join(mod_full_dir,img_mod_name) 
            # Store images
            row = pd.Series({processed_image_columns[0] :folder_logo_num,
                            processed_image_columns[1] :img_mod_name,
                            processed_image_columns[2] :file_orig_name,
                            processed_image_columns[4]: mean_mean_val,
                            processed_image_columns[5]: mean_std_val,
                            processed_image_columns[6]: mean_mscn_val,})
            
            df_logos_pickle = df_logos_pickle.append(row,ignore_index=True) 
            
           
            file_name=os.path.join(mod_full_dir,img_mod_name)
            imageio.imwrite(file_name, im_med)
            index_image =index_image+1
            brand_count_row=brand_count_row+1
            if brand_count_row > 10: # store 10 images per classes
                return df_logos_pickle
                        
   
    return df_logos_pickle

# Exclude data that doesn't satisfy the quality thresholds
def test_data_exclusion_1(df_logos_folder,dir_litw,dir_litw_resized,A,B,C):
    
    
    df_list=[]
    df_folder_Stat_list=[]
    folder_logo_num=0
    mn=50
    folders = ([name for name in sorted(os.listdir(dir_litw), key=str.casefold)]) # get all directories 
    for company_name in folders:
        contents = os.listdir(os.path.join(dir_litw,company_name)) # get list of contents
        if len(contents) <= mn: # if greater than the limit, print folder and number of contents
            continue
        company_name=company_name[4:len(company_name)] # check company name with the PD company name to retrieve stats
        pd_company_stat_row = df_logos_folder.loc[df_logos_folder["company_brand"] == company_name]
        mean_threshold=A*((pd_company_stat_row["folder_mean_val"]).values[0])
        stdev_threshold=B*((pd_company_stat_row["folder_std_val"]).values[0])
        mscn_threshold=C*((pd_company_stat_row["folder_mscn_val"]).values[0])

        folder_path_global=dir_litw
        company_name= "Mod-"+str(company_name)
        df_list.append(get_file_excluded_1(folder_path_global, company_name,dir_litw_resized, folder_logo_num, mean_threshold,stdev_threshold,mscn_threshold))

        tp_df = pd.concat(df_list, ignore_index=True)  

        folder_logo_num=folder_logo_num+1
   
    return tp_df

# Process and collect feature information
def Store_Image_Features_files(glob_path_complete_path, intermediate_layer_model, company_brand_name, folder_logo_num):
    df_feature_file_rows = pd.DataFrame(columns=feature_storage_columns)
    store_features_rows = pd.DataFrame(columns=['feature_list'])
    logo_imgs_stage = np.ndarray((1, 120, 120, 3), dtype=np.uint8)

    file_name = (os.listdir(glob_path_complete_path))

    index = 0

    features_img_rows = []
    imageID = 1
    score_folder = 0
    for record_file in file_name:
        X_train_row = []
        file_name_full = os.path.join(glob_path_complete_path, record_file)
        logo_imgs_stage = np.ndarray((1, 120, 120, 3), dtype=np.uint8)
        img_original = mpimg.imread(file_name_full)
        if imageID > 4:
            continue
        normalizedImg = cv2.normalize(img_original, None, 0, 1, cv2.NORM_MINMAX)
        logo_imgs_stage[0] = normalizedImg[:, :, :3]
        X_train_row.append(logo_imgs_stage)
        X_train = np.concatenate(X_train_row)
        feature_engg_data = intermediate_layer_model.predict(X_train)
        feature_engg_data = feature_engg_data / linalg.norm(feature_engg_data)
        data = feature_engg_data.flatten()
        row = pd.Series({feature_storage_columns[0]: folder_logo_num,
                         feature_storage_columns[1]: record_file,
                         feature_storage_columns[2]: company_brand_name,
                         feature_storage_columns[3]: imageID, })
        row_data = pd.Series({'feature_list': data, })
        df_feature_file_rows = df_feature_file_rows.append(row, ignore_index=True)
        store_features_rows = store_features_rows.append(row_data, ignore_index=True)

        imageID = imageID + 1

    return store_features_rows

# Store features in a folder for search
def Store_Image_Features_folder(dir_litw, dir_name, model):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_3').output)

    df_feature_list = []
    folder_logo_num = 0
    df_main_folder_path = []
    df_folder_list = []
    folders = ([name for name in sorted(os.listdir(dir_litw), key=str.casefold)])  # get all directories
    folder_logo_num = 0;

    for company_name in folders:
        # predict to get featured data
        glob_path_complete = Path(os.path.join(dir_litw, company_name))
        # List of features
        df_feature_list = Store_Image_Features_files(glob_path_complete, intermediate_layer_model, company_name,
                                                     folder_logo_num)
        df_feature_list_2 = df_feature_list.as_matrix()

        file_name = company_name + '.npy'
        file_name_full = os.path.join(dir_name, file_name)
        np.save(file_name_full, df_feature_list)
        df_main_folder_path.append(glob_path_complete)
        df_folder_list.append(company_name)
        folder_logo_num = folder_logo_num + 1

    # directory for storing extracted features

    np.save('output_6262019_folder2.npy', df_folder_list)
    np.save('output_6262019_folder_path2.npy', df_main_folder_path)

    return df_folder_list, df_main_folder_path, intermediate_layer_model