import os
import pandas as pd
import numpy as np
from app_logging.logger import appLogger


class DataTransform:
    """
    This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

    Written By: Anupam Hore
    Version: 1.0
    Revisions: None

    """
    def __init__(self):
        self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
        self.goodDataPath_Test = "Prediction_Raw_files_validated/Good_Raw"
        self.logger = appLogger()

    def replaceMissingWithNull(self):
        """
        Method Name: replaceMissingWithNull
        Description: This method replaces the missing values in columns with "NULL" to
                     store in the table. We are using substring in the first column to
                     keep only "Integer" data for ease up the loading.
                     This column is anyways going to be removed during training.

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        log_file = open("Training_Logs/dataTransformLog.txt", 'a+')
        try:
            onlyfiles = [f for f in os.listdir(self.goodDataPath)]
            for file in onlyfiles:
                df = pd.read_csv(self.goodDataPath + "/" + file)
                df.fillna("NULL",inplace=True)
                df.to_csv(self.goodDataPath + "/" + file, index=None,header=True)
                self.logger.log(log_file, "CSV added successfully!!")
        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
            log_file.close()
        log_file.close()

    def replaceMissingWithNull_TestData(self):
        """
        Method Name: replaceMissingWithNull_TestData
        Description: This method replaces the missing values in columns with "NULL" to
                     store in the table. We are using substring in the first column to
                     keep only "Integer" data for ease up the loading.
                     This column is anyways going to be removed during training.

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
        try:
            onlyfiles = [f for f in os.listdir(self.goodDataPath_Test)]
            for file in onlyfiles:
                df = pd.read_csv(self.goodDataPath_Test + "/" + file)
                df.fillna("NULL",inplace=True)
                df.to_csv(self.goodDataPath_Test + "/" + file, index=None,header=True)
                self.logger.log(log_file, "CSV added successfully!!")
        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
            log_file.close()
        log_file.close()


