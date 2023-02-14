"""
| **hybrid1_averaging_nlp.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to hybrid approach 1 i.e., Averaging Results of BoC, TF-IDF and TF-IDF N-Gram to Improve Accuracy for each of the ML algorithm.


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



# It stores final predictions using Naive Bayes
nb_final_verdict = []

# It stores final predictions using XGBoost
xgboost_final_verdict = []

# It stores final predictions using SVM
svm_final_verdict = []






def extract_features():
	"""
	It extracts features from training and testing datasets using BoC, TF-IDF and TF-IDF N-Gram. 
	"""
	
	logging.info('---------------------------------------------------------------------------------')
	logging.info('Features extraction using Hybrid 1 - Averaging BoC, TF-IDF and TF-IDF N-Gram Results')
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
		
		

		
	
	




def analyze_naive_bayes():
	"""
	It performs training and testing of the processed datasets with Naive Bayes algorithm.  
	"""
	
	global nb_final_verdict
	
	if (package.bag_of_characters.X_train is not None) or (package.bag_of_characters.y_train is not None) or (package.bag_of_characters.X_test is not None) or (package.bag_of_characters.y_test is not None) or (package.tf_idf.X_train is not None) or (package.tf_idf.y_train is not None) or (package.tf_idf.X_test is not None) or (package.tf_idf.y_test is not None) or (package.tf_idf_n_gram.X_train is not None) or (package.tf_idf_n_gram.y_train is not None) or (package.tf_idf_n_gram.X_test is not None) or (package.tf_idf_n_gram.y_test is not None):

		# Means features extraction has been performed already using all three NLP algorithms. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 1: Multinomial Naive Bayes')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier_boc = package.multinomial_naive_bayes.train(package.bag_of_characters.X_train, package.bag_of_characters.y_train)
		trained_classifier_tfidf = package.multinomial_naive_bayes.train(package.tf_idf.X_train, package.tf_idf.y_train)
		trained_classifier_tfidfng = package.multinomial_naive_bayes.train(package.tf_idf_n_gram.X_train, package.tf_idf_n_gram.y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		nb_predictions_boc = package.multinomial_naive_bayes.test(trained_classifier_boc, package.bag_of_characters.X_test)
		nb_predictions_tfidf = package.multinomial_naive_bayes.test(trained_classifier_tfidf, package.tf_idf.X_test)
		nb_predictions_tfidfng = package.multinomial_naive_bayes.test(trained_classifier_tfidfng, package.tf_idf_n_gram.X_test)

		for x in range(0, len(package.bag_of_characters.y_test)):
			
			if (nb_predictions_boc[x] + nb_predictions_tfidf[x] + nb_predictions_tfidfng[x]) >= 2:
				nb_final_verdict.append(1)
			else:
				nb_final_verdict.append(0)

		stop_time = time.time()
		testing_time = stop_time - start_time	
		
		#print "Final Verdict   = ", len(nb_final_verdict)
		#print "Original Labels = ", len(package.bag_of_characters.y_test)
		#print "NB BOW          = ", len(nb_predictions_boc)
		#print "NB TFIDF        = ", len(nb_predictions_tfidf)
		#print "NB TFIDF N-Gram = ", len(nb_predictions_tfidfng)


		# Analyze Predictions/Results
		package.multinomial_naive_bayes.analyze_predictions(nb_final_verdict, package.bag_of_characters.y_test)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 1 - Please first extract features before training and testing with ML algorithm')
















def analyze_xgboost():
	"""
	It performs training and testing of the processed datasets with XGBoost algorithm.  
	"""
	
	global xgboost_final_verdict
	
	if (package.bag_of_characters.X_train is not None) or (package.bag_of_characters.y_train is not None) or (package.bag_of_characters.X_test is not None) or (package.bag_of_characters.y_test is not None) or (package.tf_idf.X_train is not None) or (package.tf_idf.y_train is not None) or (package.tf_idf.X_test is not None) or (package.tf_idf.y_test is not None) or (package.tf_idf_n_gram.X_train is not None) or (package.tf_idf_n_gram.y_train is not None) or (package.tf_idf_n_gram.X_test is not None) or (package.tf_idf_n_gram.y_test is not None):

		# Means features extraction has been performed already using all three NLP algorithms. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 1: XGBoost')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier_boc = package.xgboost_model.train(package.bag_of_characters.X_train, package.bag_of_characters.y_train)
		trained_classifier_tfidf = package.xgboost_model.train(package.tf_idf.X_train, package.tf_idf.y_train)
		trained_classifier_tfidfng = package.xgboost_model.train(package.tf_idf_n_gram.X_train, package.tf_idf_n_gram.y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		xgboost_predictions_boc = package.xgboost_model.test(trained_classifier_boc, package.bag_of_characters.X_test)
		xgboost_predictions_tfidf = package.xgboost_model.test(trained_classifier_tfidf, package.tf_idf.X_test)
		xgboost_predictions_tfidfng = package.xgboost_model.test(trained_classifier_tfidfng, package.tf_idf_n_gram.X_test)

		for x in range(0, len(package.bag_of_characters.y_test)):
			
			if (xgboost_predictions_boc[x] + xgboost_predictions_tfidf[x] + xgboost_predictions_tfidfng[x]) >= 2:
				xgboost_final_verdict.append(1)
			else:
				xgboost_final_verdict.append(0)

		stop_time = time.time()
		testing_time = stop_time - start_time	
		
		
		#print "Final Verdict   = ", len(xgboost_final_verdict)
		#print "Original Labels = ", len(package.bag_of_characters.y_test)
		#print "XGBoost BOW     = ", len(xgboost_predictions_boc)
		#print "XGBoost TFIDF   = ", len(xgboost_predictions_tfidf)
		#print "XGBoost TFIDF N-Gram = ", len(xgboost_predictions_tfidfng)


		# Analyze Predictions/Results
		package.xgboost_model.analyze_predictions(xgboost_final_verdict, package.bag_of_characters.y_test)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 1 - Please first extract features before training and testing with ML algorithm')




	

		





def analyze_svm():
	"""
	It performs training and testing of the processed datasets with SVM algorithm.  
	"""
	
	global svm_final_verdict

	if (package.bag_of_characters.X_train is not None) or (package.bag_of_characters.y_train is not None) or (package.bag_of_characters.X_test is not None) or (package.bag_of_characters.y_test is not None) or (package.tf_idf.X_train is not None) or (package.tf_idf.y_train is not None) or (package.tf_idf.X_test is not None) or (package.tf_idf.y_test is not None) or (package.tf_idf_n_gram.X_train is not None) or (package.tf_idf_n_gram.y_train is not None) or (package.tf_idf_n_gram.X_test is not None) or (package.tf_idf_n_gram.y_test is not None):

		# Means features extraction has been performed already using all three NLP algorithms. 
		
		logging.info('\r---------------------------------------------------------------------------------')
		logging.info('\r Hybrid 1: Support Vector Machine (SVM)')
		logging.info('\r---------------------------------------------------------------------------------')
			
		# Perform Training
		start_time = time.time()
		trained_classifier_boc = package.svm_model.train(package.bag_of_characters.X_train, package.bag_of_characters.y_train)
		trained_classifier_tfidf = package.svm_model.train(package.tf_idf.X_train, package.tf_idf.y_train)
		trained_classifier_tfidfng = package.svm_model.train(package.tf_idf_n_gram.X_train, package.tf_idf_n_gram.y_train)
		stop_time = time.time()
		training_time = stop_time - start_time
		
		
		# Perform Testing
		start_time = time.time()
		svm_predictions_boc = package.svm_model.test(trained_classifier_boc, package.bag_of_characters.X_test)
		svm_predictions_tfidf = package.svm_model.test(trained_classifier_tfidf, package.tf_idf.X_test)
		svm_predictions_tfidfng = package.svm_model.test(trained_classifier_tfidfng, package.tf_idf_n_gram.X_test)

		for x in range(0, len(package.bag_of_characters.y_test)):
			
			if (svm_predictions_boc[x] + svm_predictions_tfidf[x] + svm_predictions_tfidfng[x]) >= 2:
				svm_final_verdict.append(1)
			else:
				svm_final_verdict.append(0)

		stop_time = time.time()
		testing_time = stop_time - start_time	
		
		
		#print "Final Verdict   = ", len(svm_final_verdict)
		#print "Original Labels = ", len(package.bag_of_characters.y_test)
		#print "SVM BOW         = ", len(svm_predictions_boc)
		#print "SVM TFIDF       = ", len(svm_predictions_tfidf)
		#print "SVM TFIDF N-Gram = ", len(svm_predictions_tfidfng)


		# Analyze Predictions/Results
		package.svm_model.analyze_predictions(svm_final_verdict, package.bag_of_characters.y_test)	
		
		package.record.add_or_update_field(field="training time", value=training_time)
		package.record.add_or_update_field(field="testing time", value=testing_time)
		
		if package.configurations.PRINT_LATENCIES.lower() == "yes":
		
			print(f"     Training Time  :{training_time}s")  
			print(f"     Testing Time   :{testing_time}s\n") 


	else:
		logging.warning('Missing step in Hybrid 1 - Please first extract features before training and testing with ML algorithm')
		
		
		
		
		
		






def all_ml_averaged():
	"""
	It performs training and testing of the processed datasets with SVM algorithm.  
	"""
	
	global nb_final_verdict, xgboost_final_verdict, svm_final_verdict

	logging.info('\r---------------------------------------------------------------------------------')
	logging.info('\r Hybrid 1: Averaging Results of All ML Algorithms')
	logging.info('\r---------------------------------------------------------------------------------')
	
		
	all_algos_final_verdict = []

	for x in range(0, len(package.bag_of_characters.y_test)):
		
		if (nb_final_verdict[x] + xgboost_final_verdict[x] + svm_final_verdict[x]) >= 2:
			all_algos_final_verdict.append(1)
		else:
			all_algos_final_verdict.append(0)


	
	# Analyze Predictions/Results
	package.svm_model.analyze_predictions(svm_final_verdict, package.bag_of_characters.y_test)	

		

	
		
		


