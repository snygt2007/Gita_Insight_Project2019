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

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```


## Configs
- A CNN based model is trained with transfer learning technique and the details of the configuration are in Image_supervised_model.py. 
- The training of model is done using %run -i Image_supervised_model.py. 


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
```

## Build Model
%run -i Image_supervised_model.py
```
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Test Results

![Input Query Image 1](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/BYD.png)

![Trademark RADAR output 1](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/byd_results.png)



![Input Query Image 2](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/starpreya.png)

![Trademark RADAR output 2](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/starpreya_results.png)



![Input Query Image 3](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/CAR_QUERY.png)

![Trademark RADAR output 3](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/CAR_output.png)



![Input Query Image 4](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/jwelery_logo.png)

![Trademark RADAR output 4](https://github.com/snygt2007/Gita_Insight_Project2019/blob/master/Readme_Images/JEWLERY_output.png)
