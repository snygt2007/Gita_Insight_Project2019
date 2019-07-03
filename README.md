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

## Initial Commit
Lets start with a blank slate: remove `.git` and re initialize the repo
```
cd $repo_name
rm -rf .git   
git init   
git status
```  
You'll see a list of file, these are files that git doesn't recognize. At this point, feel free to change the directory names to match your project. i.e. change the parent directory Insight_Project_Framework and the project directory Insight_Project_Framework:
Now commit these:
```
git add .
git commit -m "Initial commit"
git push origin $branch_name
```

## Requisites

- The packages used to build the code is provided in Requirements.txt
- Please download model from https://tinyurl.com/y2ke7stl in a models folder 

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Instructions for how to run all tests after the software is installed
```In jupyter notbook, we can pass the commands to run the code in the following manner.
# Step 1
Execute 
%run -i Raw_data_download.py
This will download all the raw logo images

# Step 2
%run -i Create_resized_data.py
This will preprocess all the data

# Step 3
%run -i Create_supervised_data.py
This will create data for supervised learning

# Step 4
%run -i Create_supervised_data.py
This will create data for supervised learning
# The training step is optional and it needs to be executed on AWS.
#Step 5
%run -i Extract_semi_supervised_data.py



## Run Final Step
#Step 6
%run -i Feature_storage.py
Dependencies:
./models folder needs to have the already trained model filename_6272019_v1_search.joblib 
query_fake folder will have the fake logo image (downloaded from internet)
Empty directory: Data_Features in the data folder
The replaced layer in the model for emdebbing creation is "dense_3"
```
# Example

# Step 1
# Step 2
```

## Build Model
%run -i Image_supervised_model.py
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

Test Results are included in the presentation slide deck for Trademark RADAR
https://preview.tinyurl.com/yytnrols
