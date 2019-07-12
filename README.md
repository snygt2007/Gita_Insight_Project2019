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

Execute 
%python online_search.py

Select a query image and clck Submit.
![Search demo](https://github.com/snygt2007/Gita_Insight_Project2019/Readme_Images/localhost_5000 - Google Chrome 7_8_2019 6_21_08PM.png)
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
