import sys
import logging
import logging.handlers as handlers

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

logHandler = handlers.RotatingFileHandler('main.log', maxBytes=5*1024*1024)  # log to the main.log file
logHandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

ch = logging.StreamHandler(sys.stdout)  # log to the stdout
ch.setFormatter(formatter)
logger.addHandler(ch)
