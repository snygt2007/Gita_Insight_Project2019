from Library_feature_storage import *

dir_litw_semisuper_clean =Path("./data/semisuper_images_cleaned")
fake_query_image_dir = Path("./data/query_fake")
resized_fake_query_dir = Path("./data/query_resized")
features_dir=Path("./data/Data_Features")
query_path=resized_fake_query_dir




model= load('./models/filename_6272019_v1_search.joblib')
model.summary()




df_folder_list,df_main_folder_path,intermediate_layer_model = Store_Image_Features_folder(dir_litw_semisuper_clean,features_dir,model)






Make_Fake_Images_Sharp(fake_query_image_dir, resized_fake_query_dir)




df_average_score_list,df_main_folder_path_new = Get_feature_score(intermediate_layer_model,query_path,features_dir)




Get_average_score_ranking(df_average_score_list,df_folder_list,df_main_folder_path_new,resized_fake_query_dir,dir_litw_semisuper_clean)






