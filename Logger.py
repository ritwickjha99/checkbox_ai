import logging
import configparser
import os
import io
from processing_module.scripts.common import keywords
from logging.handlers import TimedRotatingFileHandler
import datetime
import warnings
warnings.filterwarnings("ignore")


class Logger:
    logFilename = ""
    logLevel = ""
    # logger = None
    mainLoggerName = "CheckboxAI"
    Config = None
    job_id = None

    def readLogConfigParameter(self):
        Logger.logLevel = Logger.Config.get('Logs_Standard', 'log_level')
        filename = Logger.Config.get('Logs_Standard', 'log_filename')
        logFolderName = Logger.Config.get('Logs_Standard', 'log_dir')
        deploy_path = Logger.Config.get('Logs_Standard', keywords.INSTALLATION_DIR)
        Logger.logFilename = deploy_path + logFolderName + filename

    def getLoggerInstance(self, name=None):
        if name is None:
            log_name = self.mainLoggerName
        else:
            log_name = name

        logger = logging.getLogger(log_name)
        if not len(logger.handlers):
            self.readLogConfigParameter()
            logger = self.createLoggerInstance(self.logLevel, self.logFilename, log_name)

        return logger

    def createLoggerInstance(self, level, fileName, loggerName):
        logger = logging.getLogger(loggerName)
        hdlr = TimedRotatingFileHandler(fileName,
                                        when="D",
                                        interval=1,
                                        backupCount=5,
                                        atTime=datetime.time(11,45,00))
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)-20s - %(message)s ')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(level)
        logger.info("Log system for CheckboxAI successfully initialised")

        return logger

    def createLogConfigObject(self, deploy_path):
        with open(os.path.join(deploy_path, keywords.LOG_CONFIG_FILE)) as configFile:
            data = configFile.read()
            Logger.Config = configparser.ConfigParser()
        Logger.Config.readfp(io.StringIO(data))
        return Logger.Config