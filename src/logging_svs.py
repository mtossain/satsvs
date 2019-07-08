import sys
import logging
import logging.handlers as handlers

# logging.info('started')  # Most simple

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

logHandler = handlers.RotatingFileHandler('../output/main.log', maxBytes=5*1024*1024)  # log to the main.log file
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

ch = logging.StreamHandler(sys.stdout)  # log to the stdout
ch.setFormatter(formatter)
logger.addHandler(ch)