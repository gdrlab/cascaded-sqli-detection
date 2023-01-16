"""
| **configurations.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods for loading/updating user configurations from the configuration file. Current configurations can also be check during the application run-time.


| **CHANGE HISTORY**
| 20-03-2022        Released first version (1.0)

"""

import os.path
import sys
import logging

TRAINING_DATASET_FILE = None
TESTING_DATASET_FILE = None
PRINT_DATASET_DETAILS = None
PLOT_DATASETS = None

NGRAM_RANGE_BOW = None
NGRAM_RANGE_BOC = None
NGRAM_RANGE_TFIDF = None

PRINT_FP_FN_PAYLAODS = None
PRINT_LATENCIES = None

LogLevel = None
LogColored = None









def load(fileName, fileAddress):
	"""
	Reads input configuration file and loads the configurations. It must be called at the start of software execution so that user specified configurations are properly loaded. It also allows users to update or change configurations during software run time. For this purpose, configuration file needs to be updated and then loaded again through user command. 

	:param fileName: Name of the configuration file (e.g., nlp.conf)
	:param fileAddress: Location of the configuration file
	"""

	if os.path.isfile(fileAddress + fileName):  # Returns Ture if the file exists 
		confFile = open(fileAddress + fileName, 'r')

		try:
			lineRead = confFile.readline()

			while lineRead:
				confSettings = lineRead.split()

				if confSettings:
				
					if confSettings[0] == "TRAINING_DATASET_FILE":
						global TRAINING_DATASET_FILE 
						TRAINING_DATASET_FILE = confSettings[1]			
					elif confSettings[0] == "TESTING_DATASET_FILE":
						global TESTING_DATASET_FILE 
						TESTING_DATASET_FILE = confSettings[1]				
					elif confSettings[0] == "PRINT_DATASET_DETAILS":
						global PRINT_DATASET_DETAILS 
						PRINT_DATASET_DETAILS = confSettings[1]
					elif confSettings[0] == "PLOT_DATASETS":
						global PLOT_DATASETS 
						PLOT_DATASETS = confSettings[1]

					elif confSettings[0] == "NGRAM_RANGE_BOW":
						global NGRAM_RANGE_BOW 
						NGRAM_RANGE_BOW = confSettings[1]
					elif confSettings[0] == "NGRAM_RANGE_BOC":
						global NGRAM_RANGE_BOC 
						NGRAM_RANGE_BOC = confSettings[1]
					elif confSettings[0] == "NGRAM_RANGE_TFIDF":
						global NGRAM_RANGE_TFIDF 
						NGRAM_RANGE_TFIDF = confSettings[1]

					elif confSettings[0] == "PRINT_FP_FN_PAYLAODS":
						global PRINT_FP_FN_PAYLAODS 
						PRINT_FP_FN_PAYLAODS = confSettings[1]
					elif confSettings[0] == "PRINT_LATENCIES":
						global PRINT_LATENCIES 
						PRINT_LATENCIES = confSettings[1]


					elif confSettings[0] == "LogLevel":
						global LogLevel 
						LogLevel = confSettings[1]
					elif confSettings[0] == "LogColored":
						global LogColored 
						LogColored = confSettings[1]


				lineRead = confFile.readline()
		finally:
			confFile.close()
	else:
		logging.warning(" **** Failed to locate configuration file **** \
                                  \n   **** The software will not function correctly **** \
				  \n   **** Check if the file name and its location is correctly specified  **** \
				  \n   **** Check if the software is executed from main directory of source code  **** ")

	if TRAINING_DATASET_FILE==None or TESTING_DATASET_FILE==None or PRINT_DATASET_DETAILS==None or PLOT_DATASETS==None or NGRAM_RANGE_BOW==None or NGRAM_RANGE_BOC==None or NGRAM_RANGE_TFIDF==None or PRINT_FP_FN_PAYLAODS==None or PRINT_LATENCIES==None or LogLevel==None or LogColored==None:
		logging.warning("   **** Some or all parameters from configuration file have None value **** \
		     \n                          **** Run command 'checkconfigurations' to see incorrectly loaded configurations **** ")

	else:
		if LogLevel=="info" or LogLevel=="debug": 
			logging.info('Configurations successfully loaded')

	












def check():
	"""
	It displays/prints all of the configurations loaded from the configuration file. The values may be different if modified during the software run-time. 
	"""
	
	retMsg = "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + "               Configurations Related To Datasets                            \n"
	retMsg = retMsg + "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + " * TRAINING_DATASET_FILE  " + TRAINING_DATASET_FILE + "\n"
	retMsg = retMsg + " * TESTING_DATASET_FILE   " + TESTING_DATASET_FILE + "\n"
	retMsg = retMsg + " * PRINT_DATASET_DETAILS  " + PRINT_DATASET_DETAILS + "\n"
	retMsg = retMsg + " * PLOT_DATASETS          " + PLOT_DATASETS + "\n"
	

	retMsg = "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + "    Configurations Related To NLP based Features Extraction Algorithms  \n"
	retMsg = retMsg + "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + " * NGRAM_RANGE_BOW        " + NGRAM_RANGE_BOW + "\n"
	retMsg = retMsg + " * NGRAM_RANGE_BOC        " + NGRAM_RANGE_BOC + "\n"
	retMsg = retMsg + " * NGRAM_RANGE_TFIDF      " + NGRAM_RANGE_TFIDF + "\n"
	

	retMsg = "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + "               Configurations Related To Results                            \n"
	retMsg = retMsg + "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + " * PRINT_FP_FN_PAYLAODS   " + PRINT_FP_FN_PAYLAODS + "\n"
	retMsg = retMsg + " * PRINT_LATENCIES        " + PRINT_LATENCIES + "\n"

	
	retMsg = retMsg + "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + "               Configurations Related To Logger        \n"
	retMsg = retMsg + "---------------------------------------------------------------------------- \n"
	retMsg = retMsg + " * LogLevel               " + LogLevel  + "\n"
	retMsg = retMsg + " * LogColored             " + LogColored  + "\n"
	
	retMsg = retMsg + "----------------------------------------------------------------------------  \n\n "
	
	print(retMsg)
	
	











def update(newConfigurations, fileName, fileAddress):
	"""
	It updates configurations inside the configuration file. It allows users to update or change configurations during the software run time. For this purpose, configuration file needs to be updated and then loaded again through user command. 

	:param newConfigurations: New user specified configurations in the same format as in configuaration file (e.g., nlp.conf)
	:param fileName: Name of the configuration file (e.g., nlp.conf)
	:param fileAddress: Location of the configuration file
	"""

	# print newConfigurations
	
	allConfigWords = newConfigurations.split()
	# print "Total No. of words = ", len(newConfigurations.split())
	
	for x in range(0, len(newConfigurations.split())):
		# print "Word [%d] = " % (x), allConfigWords[x] 
		
		if allConfigWords[x].lower() == "\\n":
			parameter = allConfigWords[x+1].lower()
			newValue = allConfigWords[x+2].lower()
	
			# For python 3, use updated_Line = '{:>26}  {:>25}'.format(parameter, newValue)
			updated_Line = '%-24s  %-25s' % (parameter.upper(), newValue)

			updatedConfigurations = ""
			
			# Now open configuration file and update it.
			if os.path.isfile(fileAddress + fileName):  # Returns Ture if the file exists 
				old_confFile = open(fileAddress + fileName, 'r')
				try:
					lineRead = old_confFile.readline()

					while lineRead:
						confSettings = lineRead.split()			

						if confSettings:
							if confSettings[0].lower() == parameter:
								updatedConfigurations = updatedConfigurations + updated_Line + "\n"
							else:
								updatedConfigurations = updatedConfigurations + lineRead
						else:
							updatedConfigurations = updatedConfigurations + "\n"
						
						lineRead = old_confFile.readline()
				finally:
					old_confFile.close()	
			
			# print updatedConfigurations

			# Now open configuration file again and write updated settings.
			if os.path.isfile(fileAddress + fileName): 
				new_confFile = open(fileAddress + fileName, 'w')
							
				try:
					new_confFile.write(updatedConfigurations)
				finally:
					new_confFile.close()	
		
			logging.debug('Parameter: %s value updated inside configuration file.', parameter.upper())




