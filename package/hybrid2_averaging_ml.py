"""
| **hybrid2_averaging_ml.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to hybrid approach 2 i.e., Averaging Results of Naive Bayes, XGBoost and SVM ML algorithms for each NLP algorithm to Improve Accuracy.

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
	logging.info('Features extraction using Hybrid 2 - Averaging Naive Bayes, XGBoost and SVM Results')
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
		
		

		
		
		
		
		
	
def analyze_boc():
	"""
	It performs training and testing of the processed BoC datasets with the Naive Bayes, XGBoost and SVM algorithms and averages the results.  
	"""
		
	if (package.bag_of_characters.X_train is not None) or (package.bag_of_characters.y_train is not None) or (package.bag_of_characters.X_test is not None) or (package.bag_of_characters.y_test is not None):
		# Means features extraction has been performed already using BoC NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 2: Bag of Character (BoC)')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier_nb = package.multinomial_naive_bayes.train(package.bag_of_characters.X_train, package.bag_of_characters.y_train)
		trained_classifier_xgboost = package.xgboost_model.train(package.bag_of_characters.X_train, package.bag_of_characters.y_train)
		trained_classifier_svm = package.svm_model.train(package.bag_of_characters.X_train, package.bag_of_characters.y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		predictions_boc_nb = package.multinomial_naive_bayes.test(trained_classifier_nb, package.bag_of_characters.X_test)
		predictions_boc_xgboost = package.xgboost_model.test(trained_classifier_xgboost, package.bag_of_characters.X_test)
		predictions_boc_svm = package.svm_model.test(trained_classifier_svm, package.bag_of_characters.X_test)
		
		# It stores the final verdict on predictions 
		boc_final_verdict = []
		
		for x in range(0, len(package.bag_of_characters.y_test)):
			
			if (predictions_boc_nb[x] + predictions_boc_xgboost[x] + predictions_boc_svm[x]) >= 2:
				boc_final_verdict.append(1)
			else:
				boc_final_verdict.append(0)

		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.multinomial_naive_bayes.analyze_predictions(boc_final_verdict, package.bag_of_characters.y_test)	
		
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print("     Training Time  :{training_time}s")  
			print("     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 2 - Please first extract features before training and testing with ML algorithm')	














def analyze_tfidf():
	"""
	It performs training and testing of the processed TF-IDF datasets with the Naive Bayes, XGBoost and SVM algorithms and averages the results.  
	"""
		
	if (package.tf_idf.X_train is not None) or (package.tf_idf.y_train is not None) or (package.tf_idf.X_test is not None) or (package.tf_idf.y_test is not None):
		# Means features extraction has been performed already using TF-IDF NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 2: TF-IDF')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier_nb = package.multinomial_naive_bayes.train(package.tf_idf.X_train, package.tf_idf.y_train)
		trained_classifier_xgboost = package.xgboost_model.train(package.tf_idf.X_train, package.tf_idf.y_train)
		trained_classifier_svm = package.svm_model.train(package.tf_idf.X_train, package.tf_idf.y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		predictions_tfidf_nb = package.multinomial_naive_bayes.test(trained_classifier_nb, package.tf_idf.X_test)
		predictions_tfidf_xgboost = package.xgboost_model.test(trained_classifier_xgboost, package.tf_idf.X_test)
		predictions_tfidf_svm = package.svm_model.test(trained_classifier_svm, package.tf_idf.X_test)
		
		# It stores the final verdict on predictions 
		tfidf_final_verdict = []
		
		for x in range(0, len(package.tf_idf.y_test)):
			
			if (predictions_tfidf_nb[x] + predictions_tfidf_xgboost[x] + predictions_tfidf_svm[x]) >= 2:
				tfidf_final_verdict.append(1)
			else:
				tfidf_final_verdict.append(0)

		stop_time = time.time()
		testing_time = stop_time - start_time
		

		# Analyze Predictions/Results
		package.multinomial_naive_bayes.analyze_predictions(tfidf_final_verdict, package.tf_idf.y_test)	
		
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 2 - Please first extract features before training and testing with ML algorithm')	










def analyze_tfidf_ng():
	"""
	It performs training and testing of the processed TF-IDF N-Gram datasets with the Naive Bayes, XGBoost and SVM algorithms and averages the results.  
	"""
		
	if (package.tf_idf_n_gram.X_train is not None) or (package.tf_idf_n_gram.y_train is not None) or (package.tf_idf_n_gram.X_test is not None) or (package.tf_idf_n_gram.y_test is not None):
		# Means features extraction has been performed already using TF-IDF with N-Gram NLP algorithm. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 2: TF-IDF with N-Gram')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier_nb = package.multinomial_naive_bayes.train(package.tf_idf_n_gram.X_train, package.tf_idf_n_gram.y_train)
		trained_classifier_xgboost = package.xgboost_model.train(package.tf_idf_n_gram.X_train, package.tf_idf_n_gram.y_train)
		trained_classifier_svm = package.svm_model.train(package.tf_idf_n_gram.X_train, package.tf_idf_n_gram.y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		predictions_tfidfng_nb = package.multinomial_naive_bayes.test(trained_classifier_nb, package.tf_idf_n_gram.X_test)
		predictions_tfidfng_xgboost = package.xgboost_model.test(trained_classifier_xgboost, package.tf_idf_n_gram.X_test)
		predictions_tfidfng_svm = package.svm_model.test(trained_classifier_svm, package.tf_idf_n_gram.X_test)
		
		# It stores the final verdict on predictions 
		tfidfng_final_verdict = []
		
		for x in range(0, len(package.tf_idf_n_gram.y_test)):
			
			if (predictions_tfidfng_nb[x] + predictions_tfidfng_xgboost[x] + predictions_tfidfng_svm[x]) >= 2:
				tfidfng_final_verdict.append(1)
			else:
				tfidfng_final_verdict.append(0)

		stop_time = time.time()
		testing_time = stop_time - start_time	
		

		# Analyze Predictions/Results
		package.multinomial_naive_bayes.analyze_predictions(tfidfng_final_verdict, package.tf_idf_n_gram.y_test)	
		
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 2 - Please first extract features before training and testing with ML algorithm')
		













