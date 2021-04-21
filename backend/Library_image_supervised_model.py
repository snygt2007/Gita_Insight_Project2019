'''
This library is used to train CNN models using transfer learning technique.
The input for the library is relative path for raw image folder, processed, and cleaned image folders.
'''
#import all libraries
import os, glob
import pandas as pd
import numpy as np
import cv2, json, math, time, datetime
from pathlib import Path
from PIL import Image, ImageOps,ImageFilter
from scipy import ndimage  
import imageio 
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from joblib import dump, load
import pickle
# Use CNN InceptionV3
import keras
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

from sklearn.metrics import confusion_matrix






# Pandas column names for storing resized and preprocessed images
logo_folder_columns = ['true_logo','logo_file_name','company_brand','img_path']
cleaned_image_folders = ['true_logo','company_brand','folder_img_count','folder_path']
SAMPLE_SIZE = 124
EXPECTED_DIMENSION =120

# Get image files for training, testing
def Get_train_val_test_files(df_image_total_brand_stage,dir_supervised,company_name):
    logo_img_stage_count = len(df_image_total_brand_stage)            
    
    logo_imgs_stage = np.ndarray((logo_img_stage_count, EXPECTED_DIMENSION, EXPECTED_DIMENSION, 3), dtype=np.uint8)
    logo_img_orig_stage = np.ndarray((logo_img_stage_count, EXPECTED_DIMENSION, EXPECTED_DIMENSION, 4), dtype=np.uint8)
    y_stage = np.array(df_image_total_brand_stage['true_logo'])
    index=0
          
    for index_df, record in df_image_total_brand_stage.iterrows():
        
            image_path_info= os.path.join(dir_supervised,company_name,record["logo_file_name"])
          
        # retrieve the image and feed it into mpimg.imread
            img_get = mpimg.imread(image_path_info)
            normalizedImg = cv2.normalize(img_get,None,0,1, cv2.NORM_MINMAX)
            logo_imgs_stage[index] = img_get[:,:,:3]

        
            logo_img_orig_stage[index]=img_get
            index += 1
     
    
    return logo_imgs_stage, y_stage, logo_img_orig_stage

# Sample images for train test and validation
def Get_train_val_test_cleaned(company_list,df_filtered,dir_litw_super_clean):
    X_train_row =[]
    X_orig_train_row =[]
    X_val_row =[]
    X_orig_val_row =[]
    X_test_row=[]
    X_orig_test_row =[]
    y_train_row=[]
    y_val_row=[]
    y_test_row=[]
    
    for index_df, record in company_list.iterrows():
        
        df_image_total_brand = df_filtered[df_filtered["company_brand"] == record["company_brand"]]
        
        df_image_total_brand_train= df_image_total_brand.sample(100, replace=False, random_state=1)
        
        df_image_total_brand_test_validate = df_image_total_brand[~df_image_total_brand.index.isin(df_image_total_brand_train.index)]
        df_image_total_brand_validate = df_image_total_brand_test_validate.sample(12, replace=False, random_state=1)
        df_image_total_brand_test_inter = df_image_total_brand_test_validate[~df_image_total_brand_test_validate.index.isin(df_image_total_brand_validate.index)]
        df_image_total_brand_test = df_image_total_brand_test_inter.sample(12, replace=False, random_state=1)
        company_name=record["company_brand"]
        dir_supervised=dir_litw_super_clean
        [X_train_inter,Y_train_inter,X_orig_train_inter]=Get_train_val_test_files(df_image_total_brand_train,dir_supervised,company_name)
        X_train_row.append(X_train_inter)
        y_train_row.append(Y_train_inter)
        X_orig_train_row.append(X_orig_train_inter)
        
        
        [X_val_inter,Y_val_inter,X_orig_val_inter]=Get_train_val_test_files(df_image_total_brand_validate,dir_supervised,company_name)
        X_val_row.append(X_val_inter)
        y_val_row.append(Y_val_inter)
        X_orig_val_row.append(X_orig_val_inter)
        
        [X_test_inter,Y_test_inter,X_orig_test_inter]=Get_train_val_test_files(df_image_total_brand_test,dir_supervised,company_name)
        X_test_row.append(X_test_inter)
        y_test_row.append(Y_test_inter)
        X_orig_test_row.append(X_orig_test_inter)
      
    X_train=np.concatenate(X_train_row)
    X_orig_train=np.concatenate(X_orig_train_row)
    y_train=np.concatenate(y_train_row)
    X_val=np.concatenate(X_val_row)
    X_orig_val=np.concatenate(X_orig_val_row)
    y_val=np.concatenate(y_val_row)
    X_test=np.concatenate(X_test_row)
    y_test=np.concatenate(y_test_row)
    X_orig_test=np.concatenate(X_orig_test_row)

    return X_train,y_train,X_val,y_val,X_test,y_test,X_orig_train,X_orig_val,X_orig_test

# Get all the cleaned and processed images for supervised learning
def Get_cleaned_supervised_file_info(folder_path_global,folder_brand_name, folder_num):
    df_logos_files_supervised = pd.DataFrame(columns=logo_folder_columns)
    
     # in each folder, find image file and resize-scale them without distortion
    index_image= 1
    
    glob_path_complete_path_inter=Path(os.path.join(folder_path_global,folder_brand_name))
    brand_count_row=1
    for filename_logo in glob_path_complete_path_inter.glob('**/*.png'):
         with open(filename_logo) as imagefile:
            image_original = Image.open(filename_logo)
            dir_name,file_orig_name = os.path.split(filename_logo)
            row = pd.Series({logo_folder_columns[0] :folder_num,
                            logo_folder_columns[1] :file_orig_name,
                            logo_folder_columns[2] :folder_brand_name,
                            logo_folder_columns[3]: filename_logo,})
            
            df_logos_files_supervised = df_logos_files_supervised.append(row,ignore_index=True) 
            
            brand_count_row=brand_count_row+1
            index_image =index_image+1


    return df_logos_files_supervised

# Get brand level informartion based on sample size threshold
def Get_cleaned_supervised_folder_info(dir_litw_super):

    
    company_list = pd.DataFrame(columns=cleaned_image_folders)
    folder_logo_num=0 # y_label
    folders = ([name for name in sorted(os.listdir(dir_litw_super), key=str.casefold)]) # get all directories 
    df_list=[]
    for company_name in folders:
        contents = os.listdir(os.path.join(dir_litw_super,company_name)) # get list of contents
        if len(contents) <= SAMPLE_SIZE: # if greater than the limit, print folder and number of contents
            continue
        folder_path_global=dir_litw_super
        df_list.append(Get_cleaned_supervised_file_info(folder_path_global, company_name,folder_logo_num))
        df_image_total_brand = pd.concat(df_list, ignore_index=True) 
        
        folder_path_name=(os.path.join(folder_path_global,company_name))
        row_folder = pd.Series({cleaned_image_folders[0] :folder_logo_num,
                         cleaned_image_folders[1] :company_name,
                        cleaned_image_folders[2]: len(contents),
                        cleaned_image_folders[3]: folder_path_name,})
        
        
        company_list = company_list.append(row_folder,ignore_index=True)   
        folder_logo_num=folder_logo_num+1

    return company_list, df_image_total_brand


def print_predict_convert(X_test,y_test,model):

    predict_test_results = model.predict(X_test)
    
    predict_test_results

    y_test_1=np.array(y_test.astype(np.int64)) 
    # calculate accuracy
    y_predict = np.argmax(predict_test_results, axis=1)

    
    return(predict_test_results,y_test_1,y_test,y_predict)



# Plot the metrics
def plot_confusion_mat(y_test_1,y_predict):

    cm = confusion_matrix(y_test_1, y_predict)

    sns.set(font_scale=1.4)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d',cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Save model
def save_model(model):
    s = pickle.dumps(model)
    outfile_part = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = './models/filename_' + outfile_part + '_v1.joblib'
    dump(model, output_file)
    return
