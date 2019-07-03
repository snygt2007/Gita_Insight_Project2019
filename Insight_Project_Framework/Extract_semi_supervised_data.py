from Library_extract_semi_supervised_data import *

dir_litw=Path("./data/Raw_data")
dir_litw_resized =Path("./data/resized_images_details")
dir_litw_semisuper_clean=Path("./data/semisuper_images_cleaned")


df_logos_folder_label = pd.read_csv('.\data\Resized_folder_details.csv')
tp_df_image_resized = pd.read_csv('.\data\Resized_file_details.csv')


df_logos_folder=get_folder_stats(df_logos_folder_label,tp_df_image_resized,dir_litw_resized)



tp_df_excluded=test_data_exclusion_1(df_logos_folder,dir_litw_resized,dir_litw_semisuper_clean)




