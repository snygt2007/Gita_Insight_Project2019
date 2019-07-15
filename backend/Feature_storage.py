from backend.Library_feature_storage import *
from keras import backend as K
from keras.models import load_model
from joblib import dump, load
import pickle
import numpy as np


def search_similar_images(fake_query_image_dir, file_name):
    # The functionality of this module is to extract features from large number of classes using the embedding generator
    # This also includes a query package
    # Directory path
    model = load('./backend/models/filename_6272019_v1_search.joblib')
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_3').output)

    dir_litw_semisuper_clean =Path("./static/semisuper")
    # Directory to place the fake logo or test image

    # Query image is automatically processed
    resized_fake_query_dir = Path("./backend/data/query_resized")
    # Features are stored in a directory for speed
    features_dir=Path("./backend/data/Data_Features")
    query_path=resized_fake_query_dir
    resized_fake_dir = Path("./backend/data/query_resized")
    fake_query_image_dir = Path(fake_query_image_dir)

    # Load the embedding generator

    df_folder_list = np.load('./backend/output_6262019_folder2.npy',allow_pickle=True)
    df_main_folder_path = np.load('./backend/output_6262019_folder_path2.npy',allow_pickle=True)

    mod_query_path = Make_Fake_Images_Sharp(fake_query_image_dir, file_name,resized_fake_dir)



    df_average_score_list,df_main_folder_path_new = Get_feature_score(intermediate_layer_model,mod_query_path,features_dir)



    # Generate ranking of similar looking images
    scores_list, df_min_folder_path = Get_average_score_ranking(df_average_score_list,df_folder_list,df_main_folder_path_new,dir_litw_semisuper_clean)
    K.clear_session()
    return scores_list, df_min_folder_path




