"""
| **main.py**

| **Author:** Kasim Tasdemir 
| **Email:** -

| Version 1.0
| Date: 01-02-2023 

| **Description:**
| It helps to record the outputs of the methods in a well structured way. That would be easy to convert those results to latex tables. 

| **CHANGE HISTORY**
| 01-02-2023       Released first version 

"""


import package.configurations
import package.general_utils
import package.logger
import pandas as pd

glb_method_name = ""
dictResults = {}

def set_current_method(method_name):
    """Change the name of the current method. The table row index will be this value.

    Args:
        method_name (string): the name of the method e.g. "bow nb"
    """    
    global glb_method_name
    global dictResults
    glb_method_name = method_name
    print(f"current method name: {glb_method_name}")
    dictResults.update( {glb_method_name : {} } )

def add_or_update_field(field, value):
    global dictResults
    print(f"add or update field:{field} : {value}")
    dictResults[glb_method_name].update( {field : value} )
    print("dictResults:")
    print(dictResults)

def init():
    print("package.record.init() Initializing pandas dataframe")

def to_dataFrame():
    print("package.record.to_dataFrame()")
    return pd.DataFrame.from_dict(dictResults, orient='index')

def append():
    """
    TA
    """





