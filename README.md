# Insight_Project_Framework
Framework followed for the Trademark RADAR search engine for trademark image similarity detection.
This tool is developed for the USPTO.

## Motivation for this project format:
- **Insight_Project_Framework** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : This repository includes a query image and some input logo images that can be used to test the code.
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

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

```

## Requisites

- The packages used to build the code is provided in Requirements.txt
- Please download model from https://tinyurl.com/y2ke7stl in a models folder 
- Please obtain permission from "Tüzkö A., Herrmann C., Manger D., Beyerer J.: “Open Set Logo Detection and Retrieval“, Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications: VISAPP, 2018." to download the raw images.

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```


## Configs
- A CNN based model is trained with transfer learning technique and the details of the configuration are in Image_supervised_model.py. 




## Test
- Instructions for how to run all tests after the software is installed
python online_search.py


```
```

## Downlaod Raw data on local machine
Execute command to goto the project directory
cd backend
python download_raw_data.py 

## Preprocess data
python Create_resized_data.py
python Create_supervised_Data.py
python Extract_semi_supervised_data.py

```
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model

# Inference
![Select a query image and clck Submit.](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/data_inference.png)

# Test Results

![Images across business categories](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/business_case.png)
![Testing image](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/testing_result.png)

Changed settings
![Images across business categories](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/business_case_outline.png)
![Testing image](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/testing_outline.png)
