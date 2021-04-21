# This module is preprocessing the image files
from Library_create_resized_data import *

dir_litw=Path("./data/Raw_data")
dir_litw_resized =Path("./data/Processed_data")



df_logos_folder_label=Get_folder_name_labels(dir_litw)


tp_df_image_resized=get_all_images(dir_litw,dir_litw_resized)


# Output image files are stored
tp_df_image_resized.to_csv(r'.\data\Resized_file_details.csv')
df_logos_folder_label.to_csv(r'.\data\Resized_folder_details.csv')




