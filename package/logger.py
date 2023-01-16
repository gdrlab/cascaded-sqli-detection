"""
| **logger.py**

| **Author:** Dr. Rafiullah Khan 
| **Email:** rafiullah.khan@qub.ac.uk

| Version 1.0
| Date: 20-03-2022

| **Description:**
| This module provides methods for initializing and customizing the software logger. 


| **CHANGE HISTORY**
| 20-03-2022       Released first version (1.0)

"""



import package.configurations

import sys
import logging



# Define a Handler which writes log messages to the sys.stderr
consoleLog = logging.StreamHandler()



def add_color_to_log(function):
    """
    It changes colour of the log messages based on the log level specified in configuration file. It assigns a different colour to each log level message (debug, info, error and warning).
    """

    def color_print(*args):
        """
        Asssign a color to the message based on log level
        """
        # Just for the sake of information.
        # 0m = reset; clears all colors and styles (to white on black)
        # 1m = bold on 
        # 3m = italics on
        # 4m = underline on
        # 30m = set color to black
        # 31m = set color to red
        # 32m = set color to green
        # 33m = set color to yellow
        # 34m = set color to blue
        # 35m = set color to magenta (purple)
        # 36m = set color to cyan
        # 37m = set color to white
        # 39m = set color to default (white)
        # 40m = set background color to black
        # 47m = set background color to white
        # 49m = set background color to default (black)

        levelNum = args[1].levelno

        if(levelNum>=40):        # ERROR
                color = '\x1b[31m'
        elif(levelNum>=30):      # WARNING
                color = '\x1b[33m' 
        elif(levelNum>=20):      # INFO
                color = '\x1b[32m' 
        elif(levelNum>=10):      # DEBUG
                color = '\x1b[39m' 
        else:
                color = '\x1b[0m' 

        args[1].msg = color + args[1].msg +  '\x1b[0m'  # normal
        return function(*args)


    # Call the method color_print to change the color of message. 
    return color_print






# Initialize logging to default level 'debug' on application start
def initializeLogger():
    """
    It initializes the software logger. It must be performed at the start of software execution otherwise log messages could not be printed.
    """

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-25s %(levelname)-8s %(message)s',filename='Record.log', filemode='w')
    consoleLog.setLevel(logging.DEBUG)

    # Set a format for log messages printed on console
    formatter = logging.Formatter('%(asctime)-10s %(levelname)-8s %(message)s', datefmt='%H:%M:%S')
    # Tell the handler to use this format
    consoleLog.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(consoleLog)

    logging.debug('The logger has been initialized')




# Update logging level according to the configuration file
def updateLogger():
    """
    It updates the log based on user configurations. It defines 4 different log levels (debug, info, error and warning). User should specify his intended log level in the configuration file. Further, it also checks if the user has specified to print coloured logs and assigns different colours to each log level. Remember, each log level forbids or allows printing of different messages. For example, debug log level allows printing every type of message including information, error and warning types. Info log level prints every message except debug level. Warning log level prints only warning and error messages. Error log level only prints error messages. 
    """

    if package.configurations.LogLevel == "debug":
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)-25s %(levelname)-8s %(message)s',filename='Record.log', filemode='w')
        consoleLog.setLevel(logging.DEBUG)
    elif package.configurations.LogLevel == "info":
        logging.basicConfig(level=logging.INFO, format='%(asctime)-25s %(levelname)-8s %(message)s',filename='Record.log', filemode='w')
        consoleLog.setLevel(logging.INFO)
        logging.disable(level=logging.DEBUG)
    elif package.configurations.LogLevel == "error":
        logging.basicConfig(level=logging.ERROR, format='%(asctime)-25s %(levelname)-8s %(message)s',filename='Record.log', filemode='w')
        consoleLog.setLevel(logging.ERROR)
        logging.disable(level=logging.DEBUG)
        logging.disable(level=logging.INFO)
        logging.disable(level=logging.WARNING)
    elif package.configurations.LogLevel == "warning":
        logging.basicConfig(level=logging.WARNING, format='%(asctime)-25s %(levelname)-8s %(message)s',filename='Record.log', filemode='w')
        consoleLog.setLevel(logging.WARNING)
        logging.disable(level=logging.DEBUG)
        logging.disable(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)-25s %(levelname)-8s %(message)s',filename='Record.log', filemode='w')
        consoleLog.setLevel(logging.DEBUG)


    if package.configurations.LogColored.lower() == "yes":
        # Comment below line if not using colors for logger.
        logging.StreamHandler.emit = add_color_to_log(logging.StreamHandler.emit)

    logging.info('The logger has been updated according to the configuration file')






