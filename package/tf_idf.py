# -*- coding: utf-8 -*-

"""
| **tf_idf.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to Term Frequency — Inverse Document Frequency (TF-IDF) Approach which is an NLP algorithm to extract features from the datasets. It converts text from the datasets into matrices which can then be processed by the machine learning algorithms.


| **CHANGE HISTORY**
| 20-03-2022        Released first version (1.0)

"""

import package.datasets
import package.multinomial_naive_bayes
import package.xgboost_model
import package.svm_model

import package.configurations
import package.general_utils
import package.logger
import package.record

import logging
import time


from sklearn.feature_extraction.text import TfidfVectorizer




# Processed training dataset using TF-IDF approach
X_train = None
y_train = None

# Processed testing dataset using TF-IDF approach
X_test = None
y_test = None





# A good explanation of TF-IDF here
# https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089

# TF-IDF for characters
# https://stackoverflow.com/questions/49856775/understanding-character-level-feature-extraction-using-tfidfvectorizer

# Try N-gram with TF-IDF = Something new to try. May give better results.







def extract_features_training_dataset():
	"""
	It extracts features from training dataset using Term Frequency — Inverse Document Frequency (TF-IDF) Approach.  
	"""
	
	global X_train, y_train
		
	tf=TfidfVectorizer()
	
	# Testing Dataset: Features generation
	logging.debug('Extracting features from the training dataset.....')
	
	# Training Dataset: Features generation
	data_train_text_tf = tf.fit_transform(package.datasets.data_train['Payload'])


	X_train = data_train_text_tf
	y_train = package.datasets.data_train['Attacker'] # Labels only - attacker or benign


	if package.configurations.PRINT_DATASET_DETAILS.lower() == "yes":
		'''
		print "X_train = ", X_train
		print "y_train = ", y_train
		'''
		print(f"X_train Shape = {X_train.shape}")
		print(f"y_train Shape = {y_train.shape}")
		print(f"Features Names = {tf.get_feature_names()}")
	
	return tf














def extract_features_testing_dataset(tf):
	"""
	It extracts features from testing dataset using Term Frequency — Inverse Document Frequency (TF-IDF) Approach. 
	"""

	global X_test, y_test

	# Testing Dataset: Features generation
	logging.debug('Extracting features from the testing dataset.....')
	
	# Testing Dataset: Features generation
	#data_test_text_tf = tf.fit_transform(package.datasets.data_test['Payload'])
	data_test_text_tf = tf.transform(package.datasets.data_test['Payload'])
		
	X_test = data_test_text_tf
	y_test = package.datasets.data_test['Attacker'] # Labels only - attacker or benign		
		
	
	if package.configurations.PRINT_DATASET_DETAILS.lower() == "yes":
		'''
		print "X_test  = ", X_test
		print "y_test  = ", y_test, "\n\n"
		'''
		print(f"X_test  Shape = {X_test.shape}")
		print(f"y_test  Shape = {y_test.shape}\n\n")

















def extract_features():
	"""
	It extracts features from training and testing datasets using Term Frequency — Inverse Document Frequency (TF-IDF) Approach. 
	"""
	
	logging.info('---------------------------------------------------------------------------------')
	logging.info('Features extraction using Term Frequency — Inverse Document Frequency (TF-IDF) Approach')
	logging.info('---------------------------------------------------------------------------------')
	
	# Extract features from training dataset
	start_time = time.time()
	tf = extract_features_training_dataset()
	stop_time = time.time()
	training_dataset_processing_time = stop_time - start_time
	if package.configurations.PRINT_LATENCIES.lower() == "yes":
		logging.debug('Features extraction latency for training dataset: %f seconds', training_dataset_processing_time)


	# Extract features from testing dataset
	start_time = time.time()
	extract_features_testing_dataset(tf)
	stop_time = time.time()
	testing_dataset_processing_time = stop_time - start_time
	if package.configurations.PRINT_LATENCIES.lower() == "yes":
		logging.debug('Features extraction latency for testing dataset: %f seconds', testing_dataset_processing_time)

	
















def analyze_naive_bayes():
	"""
	It performs training and testing of the processed datasets with Naive Bayes algorithm.  
	"""
	
	global X_train, y_train, X_test, y_test

	
	if (X_train is not None) or (y_train is not None) or (X_test is not None) or (y_test is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Multinomial Naive Bayes Algorithm using TF-IDF')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier = package.multinomial_naive_bayes.train(X_train, y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		nb_predictions = package.multinomial_naive_bayes.test(trained_classifier, X_test)
		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.multinomial_naive_bayes.analyze_predictions(nb_predictions, y_test)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step - Please first extract features with TF-IDF approach before training and testing with ML algorithm')
















def analyze_xgboost():
	"""
	It performs training and testing of the processed datasets with XGBoost algorithm.  
	"""
	
	global X_train, y_train, X_test, y_test


	if (X_train is not None) or (y_train is not None) or (X_test is not None) or (y_test is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r XGBoost Algorithm using TF-IDF')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_model = package.xgboost_model.train(X_train, y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		xgboost_predictions = package.xgboost_model.test(trained_model, X_test)
		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.xgboost_model.analyze_predictions(xgboost_predictions, y_test)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n")


	else:
		logging.warning('Missing step - Please first extract features with TF-IDF approach before training and testing with ML algorithm')
		
		
		
		
		
		
		
		
		









def analyze_svm():
	"""
	It performs training and testing of the processed datasets with SVM algorithm.  
	"""
	
	global X_train, y_train, X_test, y_test


	if (X_train is not None) or (y_train is not None) or (X_test is not None) or (y_test is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r SVM Algorithm using TF-IDF')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_model = package.svm_model.train(X_train, y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		svm_predictions = package.svm_model.test(trained_model, X_test)
		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.svm_model.analyze_predictions(svm_predictions, y_test)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step - Please first extract features with TF-IDF approach before training and testing with ML algorithm')
		

















				
