"""
| **datasets.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to datasets loading, processing and analyzing. 


| **CHANGE HISTORY**
| 20-03-2022        Released first version (1.0)

"""



import package.configurations
import package.general_utils
import package.logger


# Import pandas
import pandas as pd

import logging





# Directory where datasets are stored
DATASETS_DIRECTORY = "./datasets/"


# Training dataset
data_train = None

# Testing dataset
data_test = None








def load_datasets_from_separate_files():
	"""
	It loads training and testing datasets, each from a separate file. The training and testing datasets are pre-separated or splitted in separate files. 
	"""
	
	global data_train
	global data_test

	# Load Training dataset
	logging.debug('Loading training dataset...')
	data_train = pd.read_csv(DATASETS_DIRECTORY + package.configurations.TRAINING_DATASET_FILE, sep='\t\t\t', engine='python')

	# Load Testing dataset
	logging.debug('Loading testing dataset...')
	data_test = pd.read_csv(DATASETS_DIRECTORY + package.configurations.TESTING_DATASET_FILE, sep='\t\t\t', engine='python')
	
	
	
	
	
	
	
	
	
	
	
	
def print_basic_datasets_info():
	"""
	It prints basic information about the training and testing datasets. 
	"""
	
	global data_train
	global data_test	
	
	print('--------------------------------------------------------------------------------------------------')
	print(' TRAINING DATASET: BASIC INFO')
	print('--------------------------------------------------------------------------------------------------')
	for index, count in data_train.Attacker.value_counts().iteritems():
		if index == 1:
			print(f"      Number of Attacker Samples = {count} ")
		elif index == 0:	
			print(f"      Number of Benign Samples = {count} ")
		else:
			print("      Warning: Unknown index detected in training dataset")
	print("\n") 


	print('--------------------------------------------------------------------------------------------------')
	print(' TESTING DATASET: BASIC INFO')
	print('--------------------------------------------------------------------------------------------------')
	for index, count in data_test.Attacker.value_counts().iteritems():
		if index == 1:
			print(f"      Number of Attacker Samples = {count} ")
		elif index == 0:	
			print(f"      Number of Benign Samples = {count} ")
		else:
			print(f"      Warning: Unknown index detected in training dataset")
	print(f"\n") 		












def print_detailed_datasets_info():
	"""
	It prints detailed information about the training and testing datasets. 
	"""
	
	global data_train
	global data_test	

	if package.configurations.PRINT_DATASET_DETAILS.lower() == "yes":

		print('--------------------------------------------------------------------------------------------------')
		print(' TRAINING DATASET: DETAILED INFO')
		print('--------------------------------------------------------------------------------------------------')
		print(f"Shape: \n {data_train.shape}\n\n") 
		print (f"Head: \n{data_train.head()}\n\n") 
		print (f"Info: \n{data_train.info()}\n\n") 
		print (f"Attacker Value Count: \n{data_train.Attacker.value_counts()}\n") 


		print('--------------------------------------------------------------------------------------------------')
		print(' TESTING DATASET: DETAILED INFO')
		print('--------------------------------------------------------------------------------------------------')
		print(f"Shape: \n{data_test.shape}\n\n") 
		print(f"Head: \n{data_test.head()}\n\n") 
		print(f"Info: \n{data_test.info()}\n\n") 
		print(f"Attacker Value Count: \n{data_test.Attacker.value_counts()}\n") 













def plot_datasets():
	"""
	It plots the training and testing datasets. 
	"""
	
	global data_train
	global data_test	

	if package.configurations.PLOT_DATASETS.lower() == "yes":

		# Plot training dataset
		import matplotlib.pyplot as plt
		attacker_count = data_train.groupby('Attacker').count()
		plt.bar(attacker_count.index.values, attacker_count['Payload'])
		plt.xlabel('Attacker vs Benign Samples in Training Dataset')
		plt.ylabel('Number of Samples in the training dataset')
		plt.show()


		# Plot testing dataset
		import matplotlib.pyplot as plt
		attacker_count = data_test.groupby('Attacker').count()
		plt.bar(attacker_count.index.values, attacker_count['Payload'])
		plt.xlabel('Attacker vs Benign Samples in Testing Dataset')
		plt.ylabel('Number of Samples in the testing dataset')
		plt.show()








	

