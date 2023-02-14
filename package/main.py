"""
| **main.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022 

| **Description:**
| The main file of the software. It performs different initializations, loading configurations, defining user commands, managing different threads, etc. 

| **CHANGE HISTORY**
| 20-03-2022       Released first version (1.0)

"""


import package.datasets
import package.bag_of_words
import package.bag_of_characters
import package.tf_idf
import package.tf_idf_n_gram

import package.hybrid1_averaging_nlp
import package.hybrid2_averaging_ml
import package.hybrid3_avg_best_prf
import package.hybrid4_merging_nlp_features

import package.transformers

import package.configurations
import package.general_utils
import package.logger


import logging
import threading
import time











def help():
    """
    This function prints all of the available commands and required arguments. It also provides brief description of the available commands. 
    """
    helpMessage = '-------------------------------------------------------------------------------------------------- \n'
    helpMessage = helpMessage + '  COMMAND                  DESCRIPTION \n'
    helpMessage = helpMessage + '-------------------------------------------------------------------------------------------------- \n'
    helpMessage = helpMessage + "* help:                    Display all available commands \n"
    helpMessage = helpMessage + "* all:                     It performs all software functionalities (Testing Purpose) \n"
    
    helpMessage = helpMessage + "* loaddatasets:            It loads training and testing datasets from the files \n"
    helpMessage = helpMessage + "* printdatasets:           It prints information about the training and testing datasets \n"
    helpMessage = helpMessage + "* plotdatasets:            It plots the training and testing datasets \n"
    helpMessage = helpMessage + "* extract features bow:    It extracts features from training and testing datasets using BoW approach \n"
    helpMessage = helpMessage + "* extract features boc:    It extracts features from training and testing datasets using BoC approach \n"
    helpMessage = helpMessage + "* extract features tfidf:  It extracts features from training and testing datasets using TF-IDF approach \n"
    helpMessage = helpMessage + "* extract features tfidfng:It extracts features from training and testing datasets using TF-IDF N-Gram approach \n"
    
    helpMessage = helpMessage + "* analyze bow nb:          It analyzes BoW based processed datasets using Multinomial Naive Bayes algorithm \n"
    helpMessage = helpMessage + "* analyze bow xgboost:     It analyzes BoW based processed datasets using XGBoost algorithm \n"
    helpMessage = helpMessage + "* analyze bow svm:         It analyzes BoW based processed datasets using SVM algorithm \n"
    
    helpMessage = helpMessage + "* analyze boc nb:          It analyzes BoC based processed datasets using Multinomial Naive Bayes algorithm \n"
    helpMessage = helpMessage + "* analyze boc xgboost:     It analyzes BoC based processed datasets using XGBoost algorithm \n"
    helpMessage = helpMessage + "* analyze boc svm:         It analyzes BoC based processed datasets using SVM algorithm \n"

    helpMessage = helpMessage + "* analyze tfidf nb:        It analyzes TF-IDF based processed datasets using Multinomial Naive Bayes algorithm \n"
    helpMessage = helpMessage + "* analyze tfidf xgboost:   It analyzes TF-IDF based processed datasets using XGBoost algorithm \n"
    helpMessage = helpMessage + "* analyze tfidf svm:       It analyzes TF-IDF based processed datasets using SVM algorithm \n"

    helpMessage = helpMessage + "* analyze tfidfng nb:      It analyzes TF-IDF N-Gram based processed datasets using Multinomial Naive Bayes algorithm \n"
    helpMessage = helpMessage + "* analyze tfidfng xgboost: It analyzes TF-IDF N-Gram based processed datasets using XGBoost algorithm \n"
    helpMessage = helpMessage + "* analyze tfidfng svm:     It analyzes TF-IDF N-Gram based processed datasets using SVM algorithm \n"
    
    helpMessage = helpMessage + "* hybrid1:                 Averages Results of BoC, TF-IDF and TF-IDF N-Gram to Improve Accuracy for each of the ML algorithm \n"
    helpMessage = helpMessage + "* hybrid2:                 Averages Results of Naive Bayes, XGBoost and SVM ML algorithms for each NLP algorithm to Improve Accuracy \n"
    helpMessage = helpMessage + "* hybrid3:                 Averages Results of Best Performing Algorithm for each NLP Algorithm (NB_BoC + NB_TFIDF + SVM_TFIDF_NG) \n"
    helpMessage = helpMessage + "* hybrid4:                 Merges BoC and TF-IDF N-Gram features before applying each ML algorithm \n"
            
    helpMessage = helpMessage + "* checkconfigurations:     Check the current state of configurations \n"
    helpMessage = helpMessage + "* loadconfigurations:      Load/Update configurations from file \n"
    helpMessage = helpMessage + "* updateconfigurations:    Update configurations and load them immediately. It also saves them \n"
    helpMessage = helpMessage + "                           in the configuration file. Updated configuration settings are separated \n"
    helpMessage = helpMessage + "                           by \\n and are case insensitive. Example is shown below: \n"
    helpMessage = helpMessage + "                           Example:  updateconfigurations  LogLevel  warning \\n LogColored yes \n"
    helpMessage = helpMessage + "* exit/quit:               Type exit or quit to close the software \n"
    helpMessage = helpMessage + '-------------------------------------------------------------------------------------------------- \n\n'
    
    print(helpMessage)
    
    
    











def processTheCommand(command):
    """
    It is responsible to call appropriate procedure/module based on the input command from the user.

    :param command: The command provided by the user.
    """
    command = command.lower()
    commandParts = command.split()

    if commandParts[0]=="help":
        help()


    elif commandParts[0]=="loaddatasets":
        package.datasets.load_datasets_from_separate_files()
    elif commandParts[0]=="printdatasets":
        package.datasets.print_basic_datasets_info()
        package.datasets.print_detailed_datasets_info()
    elif commandParts[0]=="plotdatasets":
        package.datasets.plot_datasets()


    elif commandParts[0]=="extract":

        if commandParts[1]=="features":

            if commandParts[2]=="bow":
                package.bag_of_words.extract_features()
                
            elif commandParts[2]=="boc":
                package.bag_of_characters.extract_features()

            elif commandParts[2]=="tfidf":
                package.tf_idf.extract_features()

            elif commandParts[2]=="tfidfng":
                package.tf_idf_n_gram.extract_features()
            
            else:
                retMsg = "Command not found!  \n"
                logging.warning(retMsg)

        else:
            retMsg = "Command not found!  \n"
            logging.warning(retMsg)


    elif commandParts[0]=="analyze":

        if commandParts[1]=="bow":

            if commandParts[2]=="nb":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.bag_of_words.analyze_naive_bayes()

            elif commandParts[2]=="xgboost":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.bag_of_words.analyze_xgboost()

            elif commandParts[2]=="svm":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.bag_of_words.analyze_svm()
                                                
            else:
                retMsg = "Command not found!  \n"
                logging.warning(retMsg)

        elif commandParts[1]=="boc":

            if commandParts[2]=="nb":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.bag_of_characters.analyze_naive_bayes()

            elif commandParts[2]=="xgboost":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.bag_of_characters.analyze_xgboost()

            elif commandParts[2]=="svm":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.bag_of_characters.analyze_svm()
                                                
            else:
                retMsg = "Command not found!  \n"
                logging.warning(retMsg)

        elif commandParts[1]=="tfidf":

            if commandParts[2]=="nb":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.tf_idf.analyze_naive_bayes()

            elif commandParts[2]=="xgboost":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.tf_idf.analyze_xgboost()

            elif commandParts[2]=="svm":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.tf_idf.analyze_svm()
                                                
            else:
                retMsg = "Command not found!  \n"
                logging.warning(retMsg)

        elif commandParts[1]=="tfidfng":

            if commandParts[2]=="nb":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.tf_idf_n_gram.analyze_naive_bayes()

            elif commandParts[2]=="xgboost":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.tf_idf_n_gram.analyze_xgboost()

            elif commandParts[2]=="svm":
                package.record.set_current_method(f"{commandParts[1]}_{commandParts[2]}" )
                package.tf_idf_n_gram.analyze_svm()
                                                
            else:
                retMsg = "Command not found!  \n"
                logging.warning(retMsg)
                                
        else:
            retMsg = "Command not found!  \n"
            logging.warning(retMsg)
            
            
    elif commandParts[0]=="hybrid1":
        package.hybrid1_averaging_nlp.extract_features()
        package.record.set_current_method(f"{commandParts[0]}_nb" )
        package.hybrid1_averaging_nlp.analyze_naive_bayes()
        package.record.set_current_method(f"{commandParts[0]}_xgboost" )
        package.hybrid1_averaging_nlp.analyze_xgboost()
        package.record.set_current_method(f"{commandParts[0]}_svm" )
        package.hybrid1_averaging_nlp.analyze_svm()
        package.record.set_current_method(f"{commandParts[0]}_all_ml_averaged" )
        package.hybrid1_averaging_nlp.all_ml_averaged()

    elif commandParts[0]=="hybrid2":
        package.hybrid2_averaging_ml.extract_features()
        package.record.set_current_method(f"{commandParts[0]}_boc" )
        package.hybrid2_averaging_ml.analyze_boc()
        package.record.set_current_method(f"{commandParts[0]}_tfidf" )
        package.hybrid2_averaging_ml.analyze_tfidf()
        package.record.set_current_method(f"{commandParts[0]}_tfidf_ng" )
        package.hybrid2_averaging_ml.analyze_tfidf_ng()

    elif commandParts[0]=="hybrid3":
        package.hybrid3_avg_best_prf.extract_features()
        package.record.set_current_method(f"{commandParts[0]}" )
        package.hybrid3_avg_best_prf.analyze_hybrid3()

            
    elif commandParts[0]=="hybrid4":
        package.hybrid4_merging_nlp_features.extract_features()
        package.record.set_current_method(f"{commandParts[0]}_nb" )
        package.hybrid4_merging_nlp_features.analyze_naive_bayes()
        package.record.set_current_method(f"{commandParts[0]}_xgboost" )
        package.hybrid4_merging_nlp_features.analyze_xgboost()
        package.record.set_current_method(f"{commandParts[0]}_svm" )
        package.hybrid4_merging_nlp_features.analyze_svm()

    elif commandParts[0]=="transformers":
        package.record.set_current_method(f"{commandParts[0]}_1" )
        package.transformers.analyze_transformers()


    elif commandParts[0]=="checkconfigurations":
        package.configurations.check()

    elif commandParts[0]=="loadconfigurations":
        package.configurations.load("nlp.conf", "./conf/")
        
    elif commandParts[0]=="updateconfigurations":
        # Update the configuration file
        package.configurations.update(command, "nlp.conf", "./conf/")
        # Now load the updated configurations       
        package.configurations.load("nlp.conf", "./conf/")


    else:
        retMsg = "Command not found!  \n"
        logging.warning(retMsg)















def software_command_loop():
    """
    This is the software command loop. It continuously waits for the user input/command and then calls the dedicated method (processTheCommand) to process. It is necessary to run this function in a separate thread to not affect the other operations of the software. 
    """
    
    package.general_utils.keep_software_running = 1
    time.sleep(0.2)
    
    print('--------------------------------------------------------------------------------------------------')
    print('                    Type \'help\' to discover all available commands                     ')
    print('--------------------------------------------------------------------------------------------------')

    while package.general_utils.keep_software_running == 1:
        if package.general_utils.is_software_initializations_performed==1:
            command = input("Enter Command > ")
            # print("Command = " + str(command))
            if command.lower()=="exit" or command.lower()=="quit":

                package.configurations.keep_software_running = 0
                time.sleep(0.5)  # Give some time for the software to stop all threads and release resources.
            
                logging.info("Closing....")
                break
            
            elif command.lower()=="all":
                # It performs all software functionalities (For Testing Purpose)
                processTheCommand("loaddatasets")
                # processTheCommand("printdatasets")
                # processTheCommand("plotdatasets")
        
                processTheCommand("extract features bow")
                processTheCommand("extract features boc")
                processTheCommand("extract features tfidf")
                processTheCommand("extract features tfidfng")
                
                processTheCommand("analyze bow nb")
                processTheCommand("analyze bow xgboost")
                processTheCommand("analyze bow svm")
                
                processTheCommand("analyze boc nb")
                processTheCommand("analyze boc xgboost")
                processTheCommand("analyze boc svm")

                processTheCommand("analyze tfidf nb")
                processTheCommand("analyze tfidf xgboost")
                processTheCommand("analyze tfidf svm")

                processTheCommand("analyze tfidfng nb")
                processTheCommand("analyze tfidfng xgboost")
                processTheCommand("analyze tfidfng svm")

                processTheCommand("hybrid1")
                processTheCommand("hybrid2")
                processTheCommand("hybrid3")
                processTheCommand("hybrid4")

                # processTheCommand("transformers")


                print(package.record.to_dataFrame())
                package.record.df_to_pickle()           
            
            elif command=="":
                continue
            else:
                processTheCommand(command)  # Software Running















def software_initalization():
    """
    This function performs basic software initialization.
    """
    
    print("----------------------------------------------------------------------------")
    
    # Initialize the logger
    package.logger.initializeLogger()

    # Load the software configurations from the configuration file.
    package.configurations.load("nlp.conf", "./conf/")

    # Update the logger based on configuration file
    package.logger.updateLogger()

    # Initialize the recorder. Creates an empty Pandas dataframe
    package.record.init()

    logging.debug('Starting command loop thread')
    threads = []
    cl_thread = threading.Thread(target=software_command_loop)
    threads.append(cl_thread)
    cl_thread.start()
    logging.info('Command loop thread successfully started')
    
    '''
    logging.debug('Starting a new thread for .....')
    my_new_thread = threading.Thread(target=my_new_thread_function)
    my_new_thread.daemon = True
    threads.append(my_new_thread)
    my_new_thread.start()
    logging.info('New thread for ....... successfully started.')
    '''
    
    
    logging.info('Completed necessary initializations...')
    package.general_utils.is_software_initializations_performed = 1





