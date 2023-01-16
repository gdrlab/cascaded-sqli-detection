"""
| **hybrid3_averaging_best_performing_ml_per_nlp.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to hybrid approach 3 i.e., Averaging Results of best performing ML algorithms (in terms of accuracy) for each NLP algorithm. Merging Results of Naive Bayes with Bag of Characters, Naive Bayes with TF-IDF and SVM with TF-IDF N-Gram.

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

import logging
import time







def extract_features():
	"""
	It extracts features from training and testing datasets using BoC, TF-IDF and TF-IDF N-Gram. 
	"""
	
	logging.info('---------------------------------------------------------------------------------')
	logging.info('Features extraction using Hybrid 3 - Best Performing Algorithms (NB_BoC + NB_TFIDF + SVM_TFIDF_NG)')
	logging.info('---------------------------------------------------------------------------------')
	
	# Extract features from training dataset
	start_time = time.time()
	boc = package.bag_of_characters.extract_features_training_dataset()
	tfidf = package.tf_idf.extract_features_training_dataset()
	tfidfng = package.tf_idf_n_gram.extract_features_training_dataset()
	stop_time = time.time()
	training_dataset_processing_time = stop_time - start_time
	if package.configurations.PRINT_LATENCIES.lower() == "yes":
		logging.debug('Features extraction latency for training dataset: %f seconds', training_dataset_processing_time)


	# Extract features from testing dataset
	start_time = time.time()
	package.bag_of_characters.extract_features_testing_dataset(boc)
	package.tf_idf.extract_features_testing_dataset(tfidf)
	package.tf_idf_n_gram.extract_features_testing_dataset(tfidfng)
	stop_time = time.time()
	testing_dataset_processing_time = stop_time - start_time
	if package.configurations.PRINT_LATENCIES.lower() == "yes":
		logging.debug('Features extraction latency for testing dataset: %f seconds', testing_dataset_processing_time)
		
		

		
		
		
		
		
	
def analyze_hybrid3():
	"""
	It performs training and testing of the processed datasets with the Best Performing Algorithms for each NLP algorithm (NB_BoC + NB_TFIDF + SVM_TFIDF_NG).  
	"""
	
	if (package.bag_of_characters.X_train is not None) or (package.bag_of_characters.y_train is not None) or (package.bag_of_characters.X_test is not None) or (package.bag_of_characters.y_test is not None) or (package.tf_idf.X_train is not None) or (package.tf_idf.y_train is not None) or (package.tf_idf.X_test is not None) or (package.tf_idf.y_test is not None) or (package.tf_idf_n_gram.X_train is not None) or (package.tf_idf_n_gram.y_train is not None) or (package.tf_idf_n_gram.X_test is not None) or (package.tf_idf_n_gram.y_test is not None):
	
		# Means features extraction has been performed already using all NLP algorithms. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 3: Averaging Best Performing Algorithms for each NLP algorithm (NB_BoC + NB_TFIDF + SVM_TFIDF_NG)')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier_nb_boc = package.multinomial_naive_bayes.train(package.bag_of_characters.X_train, package.bag_of_characters.y_train)
		trained_classifier_nb_tfidf = package.multinomial_naive_bayes.train(package.tf_idf.X_train, package.tf_idf.y_train)
		trained_classifier_svm_tfidfng = package.svm_model.train(package.tf_idf_n_gram.X_train, package.tf_idf_n_gram.y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		predictions_nb_boc = package.multinomial_naive_bayes.test(trained_classifier_nb_boc, package.bag_of_characters.X_test)
		predictions_nb_tfidf = package.multinomial_naive_bayes.test(trained_classifier_nb_tfidf, package.tf_idf.X_test)
		predictions_svm_tfidfng = package.svm_model.test(trained_classifier_svm_tfidfng, package.tf_idf_n_gram.X_test)	
		# It stores the final verdict on predictions 
		final_verdict = []
		
		for x in range(0, len(package.bag_of_characters.y_test)):
			
			if (predictions_nb_boc[x] + predictions_nb_tfidf[x] + predictions_svm_tfidfng[x]) >= 2:
				final_verdict.append(1)
			else:
				final_verdict.append(0)

		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.multinomial_naive_bayes.analyze_predictions(final_verdict, package.bag_of_characters.y_test)	
		
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 3 - Please first extract features before training and testing with ML algorithm')	









