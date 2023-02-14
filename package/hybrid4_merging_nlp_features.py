"""
| **hybrid4_merging_nlp_features.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to hybrid approach 4 i.e., Merges BoC and TF-IDF N-Gram features before applying each ML algorithm to improve accuracy.

| **CHANGE HISTORY**
| 20-03-2022        Released first version (1.0)

"""

import package.datasets
import package.multinomial_naive_bayes
import package.xgboost_model
import package.svm_model
import package.bag_of_characters
import package.tf_idf
import package.tf_idf_n_gram

import package.configurations
import package.general_utils
import package.logger

from scipy.sparse import hstack

import logging
import time




# Processed training dataset using hybrid approach 4 (Merged BoC and TF-IDF N-Gram features)
X_train_hybrid = None
y_train_hybrid = None

# Processed testing dataset using hybrid approach 4 (Merged BoC and TF-IDF N-Gram features)
X_test_hybrid = None
y_test_hybrid = None






def extract_features():
	"""
	It extracts features from training and testing datasets using BoC and TF-IDF N-Gram approaches and then merges them. 
	"""

	global X_train_hybrid, y_train_hybrid, X_test_hybrid, y_test_hybrid


	#Google "combine bag of words and tf idf features"

	# Very good link --- Check it
	# https://stackoverflow.com/questions/30653642/combining-bag-of-words-and-other-features-in-one-model-using-sklearn-and-pandas

	# Check this as well - looks great
	# https://towardsdatascience.com/multi-label-classification-using-bag-of-words-bow-and-tf-idf-4f95858740e5

	
	logging.info('---------------------------------------------------------------------------------')
	logging.info('Features extraction using Hybrid 4 - Merging BoC and TF-IDF N-Gram features together')
	logging.info('---------------------------------------------------------------------------------')
	
	# Extract features from training dataset
	start_time = time.time()
	boc = package.bag_of_characters.extract_features_training_dataset()
	tfidfng = package.tf_idf_n_gram.extract_features_training_dataset()

	# Now merge BoC and TF-IDF N-Gram features for training dataset
	# Note x_train is a sparse matrix - The technique is different for merging sparse matrices. We can use scipy.sparse.hstack to concatenate sparse matrices with the same number of rows (horizontal concatenation). We can use scipy.sparse.vstack to concatenate sparse matrices with the same number of columns (vertical concatenation). Here we need horizontal cancatenation of BoW and TF-IDF features.
	X_train_hybrid = hstack((package.bag_of_characters.X_train, package.tf_idf_n_gram.X_train))
	# y_train1 and y_train2 are same. 
	y_train_hybrid = package.bag_of_characters.y_train
	stop_time = time.time()
	training_dataset_processing_time = stop_time - start_time
	if package.configurations.PRINT_LATENCIES.lower() == "yes":
		logging.debug('Features extraction latency for training dataset: %f seconds', training_dataset_processing_time)




	# Extract features from testing dataset
	start_time = time.time()
	package.bag_of_characters.extract_features_testing_dataset(boc)
	package.tf_idf_n_gram.extract_features_testing_dataset(tfidfng)

	# Now merge BoC and TF-IDF N-Gram features for testing dataset
	X_test_hybrid = hstack((package.bag_of_characters.X_test, package.tf_idf_n_gram.X_test))
	# y_test1 and y_test2 are same.
	y_test_hybrid = package.bag_of_characters.y_test

	stop_time = time.time()
	testing_dataset_processing_time = stop_time - start_time
	if package.configurations.PRINT_LATENCIES.lower() == "yes":
		logging.debug('Features extraction latency for testing dataset: %f seconds', testing_dataset_processing_time)


	if package.configurations.PRINT_DATASET_DETAILS.lower() == "yes":

		print(f"X_train BoC Shape = {package.bag_of_characters.X_train.shape}")
		print(f"y_train BoC Shape = {package.bag_of_characters.y_train.shape}")
		print(f"X_test  BoC Shape = {package.bag_of_characters.X_test.shape}")
		print(f"y_test  BoC Shape = {package.bag_of_characters.y_test.shape}\n\n")
	
		print(f"X_train TF-IDF N-Gram Shape = {package.tf_idf_n_gram.X_train.shape}")
		print(f"y_train TF-IDF N-Gram Shape = {package.tf_idf_n_gram.y_train.shape}")
		print(f"X_test  TF-IDF N-Gram Shape = {package.tf_idf_n_gram.X_test.shape}")
		print(f"y_test  TF-IDF N-Gram Shape = {package.tf_idf_n_gram.y_test.shape}\n\n")
	
		print(f"X_train Hybrid Shape = {X_train_hybrid.shape}")
		print(f"y_train Hybrid Shape = {y_train_hybrid.shape}")
		print(f"X_test  Hybrid Shape = {X_test_hybrid.shape}")
		print(f"y_test  Hybrid Shape = {y_test_hybrid.shape}\n\n")

		#print package.bag_of_characters.X_train
		#print package.tf_idf_n_gram.X_train
		#print X_train_hybrid












def analyze_naive_bayes():
	"""
	It performs training and testing of the processed datasets with Naive Bayes algorithm.  
	"""
	
	global X_train_hybrid, y_train_hybrid, X_test_hybrid, y_test_hybrid
		
	if (X_train_hybrid is not None) or (y_train_hybrid is not None) or (X_test_hybrid is not None) or (y_test_hybrid is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 4: Multinomial Naive Bayes')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier = package.multinomial_naive_bayes.train(X_train_hybrid, y_train_hybrid)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		nb_predictions = package.multinomial_naive_bayes.test(trained_classifier, X_test_hybrid)
		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.multinomial_naive_bayes.analyze_predictions(nb_predictions, y_test_hybrid)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time, }s\n") 


	else:
		logging.warning('Missing step in Hybrid 4 - Please first extract features before training and testing with ML algorithm')
		
		
		
		
		
		
		
		
		
		
		
		
		
def analyze_xgboost():
	"""
	It performs training and testing of the processed datasets with XGBoost algorithm.  
	"""
	
	global X_train_hybrid, y_train_hybrid, X_test_hybrid, y_test_hybrid
		
	if (X_train_hybrid is not None) or (y_train_hybrid is not None) or (X_test_hybrid is not None) or (y_test_hybrid is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 4: XGBoost')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_model = package.xgboost_model.train(X_train_hybrid, y_train_hybrid)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		xgboost_predictions = package.xgboost_model.test(trained_model, X_test_hybrid)
		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.xgboost_model.analyze_predictions(xgboost_predictions, y_test_hybrid)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 4 - Please first extract features before training and testing with ML algorithm')












def analyze_svm():
	"""
	It performs training and testing of the processed datasets with SVM algorithm.  
	"""
	
	global X_train_hybrid, y_train_hybrid, X_test_hybrid, y_test_hybrid
		
	if (X_train_hybrid is not None) or (y_train_hybrid is not None) or (X_test_hybrid is not None) or (y_test_hybrid is not None):
		# Means features extraction has been performed already using an NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 4: Support Vector Machine (SVM)')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_model = package.svm_model.train(X_train_hybrid, y_train_hybrid)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		svm_predictions = package.svm_model.test(trained_model, X_test_hybrid)
		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.svm_model.analyze_predictions(svm_predictions, y_test_hybrid)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 4 - Please first extract features before training and testing with ML algorithm')
		

		






