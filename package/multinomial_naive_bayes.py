"""
| **multinomial_naive_bayes.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to Multinomial Naive Bayes algorithm such as training, testing, analyzing results, etc.


| **CHANGE HISTORY**
| 20-03-2022        Released first version (1.0)

"""

import package.datasets
import package.configurations
import package.logger

import logging


# Multinomial Naive Bayes algorithm 
from sklearn.naive_bayes import MultinomialNB

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics








def train(X_train, y_train):
	"""
	It performs training on the provided datasets using Multinomial Naive Bayes algorithm.  
	"""
	
	
	# Model Generation Using Multinomial Naive Bayes
	clf = MultinomialNB().fit(X_train, y_train)
	
	# print(clf.predict([[-0.8, -1]]))
	
	return clf
	
	
	
	


	
	
	
	
	
def test(clf, X_test):
	"""
	It performs testing on the provided datasets using Multinomial Naive Bayes algorithm.  
	"""
	
	nb_predictions = clf.predict(X_test)

	return nb_predictions
		
	
	
	
	
	
	
	


def analyze_predictions(nb_predictions, y_test):
	"""
	It analyzes results or predictions made during the testing phase and prints performance results.  
	"""
	
	#print "y_test = ", y_test          # Actual labels to payloads/data
	#print "nb_predictions = ", nb_predictions    # nb_predictions labels to payloads/data

	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0

	false_positive_payloads = []
	false_negative_payloads = []


	for x in range(0, len(y_test)):

		if (y_test[x] == 0) and (nb_predictions[x] == 0):
			true_negative = true_negative + 1
					
		elif (y_test[x] == 0) and (nb_predictions[x] == 1):
			false_positive = false_positive + 1
			false_positive_payloads.append( package.datasets.data_test.Payload[x] )
			
		elif (y_test[x] == 1) and (nb_predictions[x] == 0):
			false_negative = false_negative + 1
			false_negative_payloads.append( package.datasets.data_test.Payload[x] )
			
		elif (y_test[x] == 1) and (nb_predictions[x] == 1):
			true_positive = true_positive + 1
			
		else:
			logging.warning('Warning: Unknown outcome in results')



	overall_accuracy_percent = metrics.accuracy_score(y_test, nb_predictions)*100 
	print("     Overall Accuracy: %.2f%% " % overall_accuracy_percent) 
	
	print(f"     True Positive  :{true_positive}")
	print(f"     True Negative  :{true_negative}")
	print(f"     False Positive :{false_positive}")
	print(f"     False Negative :{false_negative}\n") 


	if package.configurations.PRINT_FP_FN_PAYLAODS.lower() == "yes":
		
		print("False Positive Payloads:")		
		for payload in false_positive_payloads:
			print(f"        {payload}")		


		print("\nFalse Negative Payloads:")		
		for payload in false_negative_payloads:
			print(f"        {payload}")	














