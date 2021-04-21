from Library_extract_semi_supervised_data import *
from joblib import dump, load



# The functionality of this module is to provide automated good quality data to provide large number of classes for semi-supervised learning
# Directory path
dir_litw=Path("./data/Raw_data")
dir_litw_resized =Path("./data/Processed_data")
dir_litw_semisuper_clean=Path("../static/Semisuper")
features_dir=Path("./data/Data_Features")
# Automated cleaning hyperparameters for semi-supervised learning
A=1.0
B=1.0
C=0.8
# Output storage
df_logos_folder_label = pd.read_csv('.\data\Resized_folder_details.csv')
tp_df_image_resized = pd.read_csv('.\data\Resized_file_details.csv')


df_logos_folder=get_folder_stats(df_logos_folder_label,tp_df_image_resized,dir_litw_resized)



tp_df_excluded=test_data_exclusion_1(df_logos_folder,dir_litw_resized,dir_litw_semisuper_clean,A,B,C)

# Load the embedding generator
model = load('./models/filename_6272019_v1_search.joblib')
model.summary()

df_folder_list, df_main_folder_path, intermediate_layer_model = Store_Image_Features_folder(dir_litw_semisuper_clean,
                                                                                            features_dir, model)

