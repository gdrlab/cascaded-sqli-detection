# -*- coding: utf-8 -*-

"""
| **bag_of_words.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to Bag of Words (BoW) Approach which is an NLP algorithm to extract features from the datasets. It converts text from the datasets into matrices which can then be processed by the machine learning algorithms.


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


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer




# Processed training dataset using BoW approach
X_train = None
y_train = None

# Processed testing dataset using BoW approach
X_test = None
y_test = None












# Bag-of-words model (BoW) is the simplest way of extracting features from the text. BoW converts text into the matrix of occurrence of words within a document. This model concerns about whether given words occurred or not in the document.
# https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# It is using n-gram appraoch





def extract_features_training_dataset():
	"""
	It extracts features from training dataset using Bag of Words (BoW) Approach.  
	"""
	
	global X_train, y_train
		
	
	# Tokenizer to remove unwanted elements from out data like symbols and numbers
	token = RegexpTokenizer(r'[a-zA-Z0-9]+')


	# Word based n-gram or character-based n gram.
	# analyzer{‘word’, ‘char’, ‘char_wb’} or callable, default=’word’
	# Details Here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
	#cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
	cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,int(package.configurations.NGRAM_RANGE_BOW)),tokenizer = token.tokenize)


	# Training Dataset: Features generation
	logging.debug('Extracting features from the training dataset.....')
	data_train_text_counts = cv.fit_transform(package.datasets.data_train['Payload'])
	#data_train_text_counts = cv.transform(package.datasets.data_train['Payload'])

	#print "cv = ", cv
	#print "data_train_text_counts = ", data_train_text_counts

	X_train = data_train_text_counts
	y_train = package.datasets.data_train['Attacker'] # Labels only - attacker or benign



	if package.configurations.PRINT_DATASET_DETAILS.lower() == "yes":
		'''
		print "X_train = ", X_train
		print "y_train = ", y_train
		'''
		print (f"X_train Shape = {X_train.shape}")
		print (f"y_train Shape = {y_train.shape}")


	return cv










def extract_features_testing_dataset(cv):
	"""
	It extracts features from testing dataset using Bag of Words (BoW) Approach.  
	"""

	global X_test, y_test

	# Testing Dataset: Features generation
	logging.debug('Extracting features from the testing dataset.....')
	#data_test_text_counts = cv.fit_transform(package.datasets.data_test['Payload'])
	data_test_text_counts = cv.transform(package.datasets.data_test['Payload'])


	#print "cv = ", cv
	#print "data_test_text_counts  = ", data_test_text_counts

	X_test = data_test_text_counts
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
	It extracts features from training and testing datasets using Bag of Words (BoW) Approach.  
	"""
	
	logging.info('---------------------------------------------------------------------------------')
	logging.info('Features extraction using Bag of Words (BoW) Approach')
	logging.info('---------------------------------------------------------------------------------')
	
	# Extract features from training dataset
	start_time = time.time()
	cv = extract_features_training_dataset()
	stop_time = time.time()
	training_dataset_processing_time = stop_time - start_time
	if package.configurations.PRINT_LATENCIES.lower() == "yes":
		logging.debug('Features extraction latency for training dataset: %f seconds', training_dataset_processing_time)


	# Extract features from testing dataset
	start_time = time.time()
	extract_features_testing_dataset(cv)
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
		logging.info('\rMultinomial Naive Bayes Algorithm using Bag of Words (BoW)')
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
		logging.warning('Missing step - Please first extract features with BoW approach before training and testing with ML algorithm')
		
		
		
		
		







def analyze_xgboost():
	"""
	It performs training and testing of the processed datasets with XGBoost algorithm.  
	"""
	
	global X_train, y_train, X_test, y_test


	if (X_train is not None) or (y_train is not None) or (X_test is not None) or (y_test is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r XGBoost Algorithm using Bag of Words (BoW)')
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
		logging.warning('Missing step - Please first extract features with BoC approach before training and testing with ML algorithm')









		
		
		
		



def analyze_svm():
	"""
	It performs training and testing of the processed datasets with SVM algorithm.  
	"""
	
	global X_train, y_train, X_test, y_test


	if (X_train is not None) or (y_train is not None) or (X_test is not None) or (y_test is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r SVM Algorithm using Bag of Words (BoW)')
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
		
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step - Please first extract features with BoC approach before training and testing with ML algorithm')
















		
