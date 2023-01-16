"""
| **svm_model.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods related to Support Vector Machine (SVM) algorithm such as training, testing, analyzing results, etc.


| **CHANGE HISTORY**
| 20-03-2022        Released first version (1.0)

"""

import package.datasets
import package.configurations
import package.logger

import logging


from sklearn import svm
from sklearn.metrics import accuracy_score


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics








def train(X_train, y_train):
	"""
	It performs training on the provided datasets using SVM algorithm.  
	"""
	
	# Classifier - Algorithm - SVM
	# fit the training dataset on the classifier
	trained_model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	trained_model.fit(X_train,y_train)
		
	return trained_model
	
	
	
	
	
	
	
	
	
	
def test(trained_model, X_test):
	"""
	It performs testing on the provided datasets using SVM algorithm.  
	"""

	# predict the labels on validation dataset
	svm_predictions = trained_model.predict(X_test)

	return svm_predictions
		
	
	












def analyze_predictions(svm_predictions, y_test):
	"""
	It analyzes results or predictions made during the testing phase and prints performance results.  
	"""
	
	#print "y_test = ", y_test          # Actual labels to payloads/data
	#print "svm_predictions = ", svm_predictions    # svm_predictions labels to payloads/data

	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0

	false_positive_payloads = []
	false_negative_payloads = []


	for x in range(0, len(y_test)):

		if (y_test[x] == 0) and (svm_predictions[x] == 0):
			true_negative = true_negative + 1
					
		elif (y_test[x] == 0) and (svm_predictions[x] == 1):
			false_positive = false_positive + 1
			false_positive_payloads.append( package.datasets.data_test.Payload[x] )
			
		elif (y_test[x] == 1) and (svm_predictions[x] == 0):
			false_negative = false_negative + 1
			false_negative_payloads.append( package.datasets.data_test.Payload[x] )
			
		elif (y_test[x] == 1) and (svm_predictions[x] == 1):
			true_positive = true_positive + 1
			
		else:
			print("Warning: Unknown outcome in results"	)



	overall_accuracy_percent = metrics.accuracy_score(y_test, svm_predictions)*100 
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










