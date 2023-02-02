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
import os
from datetime import datetime

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

glb_method_name = ""
dictResults = {}
dfResults = pd.DataFrame({'A' : []})

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
    #print(f"add or update field:{field} : {value}")
    dictResults[glb_method_name].update( {field : value} )
    #print("dictResults:")
    #print(dictResults)

def init():
    print("package.record.init() Initializing pandas dataframe")

def to_dataFrame():
    print("package.record.to_dataFrame()")
    global dfResults
    dfResults = pd.DataFrame.from_dict(dictResults, orient='index')
    return dfResults

def df_to_pickle():
    print("package.record.df_to_pickle()")
    
    currentDateAndTime = datetime.now()
    cwd = os.getcwd()
    currentTime = currentDateAndTime.strftime("%y%m%d_%H%M%S")
    output_filename = f"results_{currentTime}.pkl"
    print(f"Saving results dataframe as pandas pickle to : {cwd}/{output_filename}" ) 
    dfResults.to_pickle(f"./{output_filename}")
    print("Saved.")

def metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    package.record.add_or_update_field("precision", precision)
    package.record.add_or_update_field("recall", recall)
    package.record.add_or_update_field("f1", f1)
    #package.record.add_or_update_field("support", support)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    package.record.add_or_update_field("tn", tn)
    package.record.add_or_update_field("fp", fp)
    package.record.add_or_update_field("fn", fn)
    package.record.add_or_update_field("tp", tp)





