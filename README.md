# Hybrid Features Extraction Approach using Natural Language Processing for Improved Detection of SQL Injection Attack

This is the implementation of "Hybrid Features Extraction Approach using Natural Language Processing for Improved Detection of SQL Injection Attack" algorithm. It can load the custom datasets, train various models and demonstrate their inference performances. 
## Requirements
- [Anaconda](https://www.anaconda.com/products/distribution) environment
## Setup and Installation
It can run on virtual Python environment on Win, Mac or Linux.

- Download and install the [Anaconda](https://www.anaconda.com/products/distribution) environment.
- Open Anaconda Prompt and change current directory to the project folder. 
- Run the following command in the folder, where **rafi-sqli.yml** file resides. This will create a new Python environment with the required packages:
    -  ``` conda env create -f rafi-sqli.yml ```
- Activate the environment
    - ``` conda activate rafi-sql ```
- Build and install the **nlp_hybrid** package by running:
    - ``` python setup.py install ```
- Make sure that the *train* and *test* datasets are in the Dataset folder, and conf/nlp.conf file is modified correctly.
- Then run the nlp_hybrid library interface:
    - ``` nlp_hybrid ```
    - Finally run all tests by (or type *help* for other commands):
    - ``` all ```

 
