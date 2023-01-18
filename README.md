# Hybrid Features Extraction Approach using Natural Language Processing for Improved Detection of SQL Injection Attack

This is the implementation of "Hybrid Features Extraction Approach using Natural Language Processing for Improved Detection of SQL Injection Attack" algorithm. It can load the custom datasets, train various models and demonstrate their inference performances. 
## Requirements
- [Anaconda](https://www.anaconda.com/products/distribution) environment
## Setup
It can run on virtual Python environment on Win, Mac or Linux.

- Download and install the [Anaconda](https://www.anaconda.com/products/distribution) environment.
- Open Anaconda Prompt and change current directory to the project folder. 
- Run the following command in the folder, where **rafi-sqli.yml** file resides. This will create a new Python environment with the required packages:
    -  ``` conda env create -f rafi-sqli.yml ```
- Activate the environment
    - ``` conda activate rafi-sql ```

### (Optional) Build and install as a package
- Build and install the **nlp_hybrid** package by running:
    - ``` python setup.py install ```

## Running
- Make sure that the *train* and *test* datasets are in the Dataset folder, and conf/nlp.conf file is modified correctly.

There are two ways to run the code. It can be directly executed as a python script without building as a package. Or, it can be build, installed and run from the console.

- If you haven't built as a package, run:
    - ``` python nlp_hybrid.py ```
- (Optional)If you have built and installed the package, then run the nlp_hybrid library interface:
    - ``` nlp_hybrid ```


- Finally, to run all tests by (or type *help* for other commands):
    - ``` all ```
## Troubleshot
- if something goes wrong or you change the code, clean the installed folders and rebuild it:
    - ``` python setup.py clean --all ```
## Release notes
- Release (Under development - The current code)

    - The original code required datasets with 'delimiter=three tabs'. This is no longer supported by Pandas data frame. So, It has been changed to support single tab delimited dataset. If you need to use the old code on the old datasets, you can use Release v0.0.12
	- utils/clean-kaggle-sqli-dataset-and-split.ipynb file is created for cleaning Kaggle SQLi dataset and splitting it into train-test files.
	- utils/convert-old-dataset-to-new-single-tab.ipynb file is created for converting old three tabs delimited dataset to single tab.
	- utils/dataset-train-test-splitter.ipynb file is created for splitting the given dataset to train and test parts.
	- support for running without building the package is added (nlp_hybrid.py)

- Release v0.0.12

    - This is the original , the first code from Rafi. It works with Python 3 (the very first one was Python 2)


 
