# This module prepares data for supervised learning
from Library_create_supervised_data import *



dir_litw_resized=Path("./data/Processed_data")
dir_litw =Path("./data/Raw_data")
dir_litw_super_clean=Path("./data/Supervised_data")


df_logos_folder_label = pd.read_csv('.\data\Resized_folder_details.csv')
tp_df_image_resized = pd.read_csv('.\data\Resized_file_details.csv')




df_logos_folder_label.head()
tp_df_image_resized.head()


df_logos_folder=get_folder_stats(df_logos_folder_label, tp_df_image_resized,dir_litw_resized)



A=0.5
B=0.5
C=0.4
# Customized automated filtering using statistical tools
tp_df_excluded=test_data_exclusion_1(df_logos_folder,A,B,C, dir_litw_super_clean,dir_litw_resized)



Print_filtered_folder_details(dir_litw_super_clean)


tp_df_image_resized.to_pickle('image_logos_folder_path_7022019.pkl')





