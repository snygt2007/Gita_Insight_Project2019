# Overview
Python search engine tool for detecting similar trademarks. Pipeline below for detecting trademark infringement for an institution like US Patent Office

![Trademark RADAR Demo](static/readme_images/git_demo_v7.gif)

## Motivation for this project format:
- **backend** : Put all source code for production within structured directory. It contains model, data, and code
Attribution: 
https://live.ece.utexas.edu/publications/2011/am_asilomar_2011.pdf 
https://github.com/ilmonteux/logohunter (Logo images)
https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/


- **tests** : Put all source code for testing in an easy to find location
- **static** : Any images or content to include in the web framework
- **templates** : Flask web page templates
- **online_search.py** : This maps the web page requests to the backend modules

## Setup
Clone repository and update python path
```
repo_name=Gita_Insight_Project2019 # URL of your new repository
username=snygt2007 # Username for your personal github account
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
Create new development branch and switch onto it
```
branch_name=dev-readme_requisites-20180905 # Name of development branch, of the form 'dev-feature_name-date_of_creation'}}
git checkout -b $branch_name
```


## Prerequisites

- The packages used to build the code is provided in Requirements.txt
- Please download model from https://tinyurl.com/y2ke7stl in backend/models folder 
- Please obtain permission from "Tüzkö A., Herrmann C., Manger D., Beyerer J.: “Open Set Logo Detection and Retrieval“, Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications: VISAPP, 2018." to download the raw images and perform automated processing and training. Alternatively, you can execute the inference portions.

#### Installation
To install the package above, please run:
```shell
pip install -r requiremnts.txt
```


## Test
- Instructions for how to run all tests after the software is installed
(This requires the model to be downloaded in the backend/models folder)
python online_search.py
start localhost:5000 in your local browser (google chrome)
Select a png image for search query. Alternatively, you can select an existing image from the static/semisuper to execute the search.
```
```

## Download Raw data on local machine
Execute command to goto the project directory
cd backend
python download_raw_data.py 

## Preprocess data
python Create_resized_data.py
python Create_supervised_Data.py



## Configs
- A CNN based model is trained with transfer learning technique and the details of the configuration are in 
python Image_supervised_model.py. 

- Features from the existing logo catalogs are extracted and stored using
python Extract_semi_supervised_data.py

# Inference
![Select a query image and clck Submit.](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/data_inference.png)

# Test Results

![Images across business categories](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/business_case.png)
![Testing image](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/testing_result.png)

![Changed settings]
![Images across business categories](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/business_case_outline.png)
![Testing image](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/testing_outline.png)
