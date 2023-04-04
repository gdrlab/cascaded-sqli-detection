# Hybrid Features Extraction Approach using Natural Language Processing for Improved Detection of SQL Injection Attack

This is the implementation of "Hybrid Features Extraction Approach using Natural Language Processing for Improved Detection of SQL Injection Attack" algorithm. It can load the custom datasets, train various models and demonstrate their inference performances. 
## Requirements  

- (recommended) Google Colab account, or any other Jupyter Notebook with GPU support.
- (optional) If you want to run on local machines, use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge). Mamba package search is significantly faster than Anaconda (or [Anaconda](https://www.anaconda.com/products/distribution) environment)
## Setup
### For Google Colab setup
- Upload all GitHub files into your Google Drive (e.g. '/content/drive/MyDrive/Akademik/Research and Projects/Sakir Hoca Projects/AI Security Intelligence/Codes/20230331_sqli_colab')
- Update the hardcoded paths and run 'main.ipynb'
### (optional) For local setup 
You can skip this part if you use Google Colab. 

The first part (Classical ML methods) can run on virtual Python environment on Win, Mac or Linux.

- Download and install the [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) environment.
- Open Miniforge Prompt and change current directory to the project folder. 
- (Optional) If you have GPU and want to use it update the yml file to contain "tensorflow-gpu".(this might not work. Not tested)
- Run the following command in the folder, where **rafi-sqli.yml** file resides. This will create a new Python environment with the required packages:
    -  ``` mamba env create -f rafi-sqli.yml ```
- Activate the environment
    - ``` mamba activate rafi-sqli ```
- Install the following packages:
    - ``` pip install -q -U "tensorflow-text==2.8.*" ```
    - ``` pip install -q tf-models-official==2.7.0 ```
    - (Note: these packages were not in conda repo. If you find it in the repo, use mamba install instead of pip)


## Running
### Running all
- Modify config.ini file and run 'main.ipynb'

### (Optional) Running only the classical ML based methods
- Activate rafi-sqli environment in the Miniforge prompt and run the test:
    - ``` python run_classical_MLs.py -o <output file path>```


## Demonstrating the experimental results

- Modify ``` utils\Demonstrate_test_results.ipynb ``` to point the result file and run it.

## Folder contents
- Main folders: (datasets, utils, trained_models, results)
  - datasets: SQLi csv file with two columns:'payload' and 'label'.
  - config.ini: choose the models to be tested and other options. Note: Ensemble models need all classical MLs to be run before.
  - utils   
    - Demonstrate test results : produce all visuals and tables used in the paper.

## Troubleshot

- 
## Release notes
- Release (vTBA)
  - Supports multiple random seeds in config.ini file.
- Release (v0.4.0)
  - OOP is used to modularize the code.
  - Most of the settings can be set from config.ini file.
  - Classic ML and Transformers can be run with a single file (notebook)
  - "results demonstration" file is updated
  - results are saved into CSV file, not pkl.
  - "saving models" does NOT work.
- Release (0.3.0) Latex table generation
  - The results are saved to a pkl file in the main folder.
  - the results pickle files can be read and visualized using utils\Demonstrate_test_results.ipynb. This also generates the Latex tables used in the paper.
  - the visualized results have color scheme. The fonts sizes, etc. are ready for the paper.
- Release (v0.2.0)
    - BERT is added to the tests. See ``` package/detect_sqi_with_transformers.ipynb ```
    - ``` utils\Demonstrate_test_results.ipynb ``` combine the results and demonstrates the experimental results.
- Release (v0.1.0)
    - Results are saved to a Pandas dataframe. It is saved to a pickle file.
    - Results can be visualized using Utils/Data visualize . jpy notebook.
    - The original code required datasets with 'delimiter=three tabs'. This is no longer supported by Pandas data frame. So, It has been changed to support single tab delimited dataset. If you need to use the old code on the old datasets, you can use Release v0.0.12
	- utils/clean-kaggle-sqli-dataset-and-split.ipynb file is created for cleaning Kaggle SQLi dataset and splitting it into train-test files.
	- utils/convert-old-dataset-to-new-single-tab.ipynb file is created for converting old three tabs delimited dataset to single tab.
	- utils/dataset-train-test-splitter.ipynb file is created for splitting the given dataset to train and test parts.
	- support for running without building the package is added (nlp_hybrid.py)

- Release v0.0.12

    - This is the original , the first code from Rafi. It works with Python 3 (the very first one was Python 2)

## TODO
- Save the trained models. Add a method to load and run the saved models without training.


 
