'''
This library is used to search similar images from the catalog for a given query image. 
Images are represented in embedded feature space.
'''
#import all libraries
import os, glob
import pandas as pd
import numpy as np
import cv2, csv, json, math, time
from pathlib import Path
from PIL import Image, ImageOps,ImageFilter
from scipy import ndimage  
import imageio 
from pylab import *
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import pickle
import h5py
import matplotlib.image as mpimg




SEMI_PATH = '/static/semisuper/'

#Image processing for resizing images
def fix_aspect_ratio(img,expected_dimension):
    original_max_dim = max(float(img.size[0]),float(img.size[1]))
    original_min_dim = min(float(img.size[0]),float(img.size[1]))
    wpercent = (expected_dimension/float(original_max_dim))
    hsize = int((original_min_dim*float(wpercent)))
    new_im = img.resize((expected_dimension,hsize), Image.ANTIALIAS)
    return new_im

# Make square images
def make_square(img, min_size=120, fill_color=(0, 0, 0, 0)):
    x, y = img.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


# Sharpen the edges
from pylab import *
def sharpen_filter(img):
    sharp_im = img.filter(ImageFilter.SHARPEN)
    return sharp_im


# MSCN value
def calculate_MSCN_val(img):
    C=3.0/255.0
    blurred_img=cv2.GaussianBlur(img, (7, 7), 1.166)
    blurred_sq = blurred_img * blurred_img 
    sigma = cv2.GaussianBlur(img * img, (7, 7), 1.166) 
    sigma = (sigma - blurred_sq) ** 0.5
    sigma = sigma + C
    MCSN_value = (img - blurred_img)/sigma
    return MCSN_value



# Process incoming query image in the same way as the catalog images
def Make_Fake_Images_Sharp(glob_path_complete_path, file_name,resized_fake_dir):

    expected_dimension=120
    index_image =1
    filename_logo = os.path.join(glob_path_complete_path,file_name)
    image_original = Image.open(filename_logo)
    img_mod_name = 'Fake-AB-' + str(index_image) + '.png'

    os.makedirs(resized_fake_dir, exist_ok=True)
    path_to_image = os.path.join(resized_fake_dir, img_mod_name)

    image_aspect=fix_aspect_ratio(image_original,expected_dimension)

    img_new = make_square(image_aspect)

    sharpen_image=sharpen_filter(img_new)
    im_med = ndimage.median_filter(sharpen_image, 3)
    file_name = os.path.join(resized_fake_dir, img_mod_name)
    imageio.imwrite(file_name, im_med)

    return file_name

def Extract_query_img_feature(dir_litw,intermediate_layer_model):
    

    df_list=[]
    imageID=1
 
    index = 0
    X_train_row =[]
    logo_imgs_stage = np.ndarray((1, 120, 120, 3), dtype=np.uint8)
    glob_path_complete_path = (dir_litw)
    file_name = glob_path_complete_path
    img_original = mpimg.imread(file_name)
    normalizedImg = cv2.normalize(img_original,None,0,1, cv2.NORM_MINMAX)
    logo_imgs_stage[index] = normalizedImg[:,:,:3]
    X_train_row.append(logo_imgs_stage)
    X_train=np.concatenate(X_train_row)
    feature_engg_data = intermediate_layer_model.predict(X_train)


    imageID = imageID+1
    return  feature_engg_data




def cosine_distance(query_feature, stored_feature):
    stored_feature = stored_feature.reshape(1, -1)
    #print(stored_feature)
    query_feature = query_feature.reshape(1, -1)
    #print(query_feature)
    result=cosine_similarity(stored_feature,query_feature)
    #result = 1 - spatial.distance.cosine(stored_feature, query_feature)
    return result[0][0]
# Get average score per brand
def Get_average_score_ranking(df_average_score_list,df_folder_list,df_main_folder_path, dir_litw_semisuper_clean):
    
    scores = np.array(df_average_score_list)
   
    rank_ID = np.argsort(scores)[:-11:-1]
    rank_score = scores[rank_ID]

    maxres=10

    im_file_list = []
    
    df_main_folder_path = [(df_main_folder_path[index_2]) for i_2,index_2 in enumerate(rank_ID[0:maxres])]
    scores_list = [round((scores[index_3]),3) for i_3,index_3 in enumerate(rank_ID[0:maxres])]

    for i,im_folder_name in enumerate(df_main_folder_path):
        print(df_main_folder_path[i])
        mod_folder_name_head,mod_folder_name_tail = os.path.split(im_folder_name)
        length_folder = len(mod_folder_name_tail)-4
        mod_folder_name = mod_folder_name_tail[:length_folder]
        dir_litw_folder = os.path.join(dir_litw_semisuper_clean,mod_folder_name)


        im_file=os.listdir(dir_litw_folder)

        im_path_file_full=os.path.join(dir_litw_folder,im_file[0])
 
        im_path_file_full = SEMI_PATH + mod_folder_name + '/' + im_file[0]
        im_file_list.append(im_path_file_full)


    return scores_list, im_file_list

# Use cosine similarity for ranking images
def Get_Image_Features_files(file_record,intermediate_layer_model,query_feature):
    logo_imgs_stage = np.ndarray((1, 120, 120, 3), dtype=np.uint8)
    
    
    length_file = len(file_record)

    i_index = 0
   
    record_file_rows=[]
    features_img_rows=[]
    imageID=1

    score_folder=0
    for index in range(length_file):
        X_train_row =[]
        
        feature_engg_data = (file_record[i_index][0])

        score_file=cosine_distance(query_feature, feature_engg_data.T) 
       
        score_folder=score_folder+score_file
                
        i_index = i_index+1
        
    average_score= score_folder/length_file
    
    return average_score

# Get score of each image vector
def Get_feature_score(intermediate_layer_model,mod_query_path, features_path):

    dir_litw_features =Path(features_path)
   

    df_average_score_list=[]
    folder_logo_num=0
    df_main_folder_path=[]
    
    File_names = ([name for name in sorted(os.listdir(dir_litw_features), key=str.casefold)]) # get all directories 
    
    folder_logo_num=0
    query_features=Extract_query_img_feature(mod_query_path,intermediate_layer_model)
    for file_features_list in File_names:
    #predict to get featured data
        if file_features_list == '.ipynb_checkpoints':
            continue
        File_name_full = os.path.join(dir_litw_features,file_features_list)
        file_features_record = np.load(File_name_full, allow_pickle=True)
        
        df_average_score_list.append(Get_Image_Features_files(file_features_record,intermediate_layer_model,query_features))
        df_main_folder_path.append(File_name_full)
        folder_logo_num=folder_logo_num+1
        
    return df_average_score_list,df_main_folder_path
