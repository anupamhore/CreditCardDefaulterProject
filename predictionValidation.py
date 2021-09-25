import os
from app_logging.logger import appLogger
from RawDataValidation.testdatavalidation import TestDataValidation
from DataTransform_Training.DataTransform import DataTransform
from DBOps.dbOperationsforPrediction import DBOperationsPrediction

class predictValidation:
    """
    Class Name: predictValidation
    Description: This class shall be used for validation of the test dataset. We will check
                 the number of variables,and the format of the variables in the dataset are equal
                 to the agreed number.We will impute any missing values to NULL values so that it
                 could be properly inserted into the Cassandra DB and later can be downloaded from
                 the db for testing purpose

    Written By: Anupam Hore
    Version: 1.0
    Revisions: None

    """

    def __init__(self,file):
        self.cwd = os.getcwd()
        self.file_object = open("Prediction_Logs/Prediction_Main_Log.txt", 'a+')
#         self.file_object = open(self.cwd + 'Prediction_Main_Log.txt', 'a+')
        self.log_writer = appLogger()
        self.raw_data = TestDataValidation(file)
        self.dataTransform = DataTransform()
        self.dbOperation = DBOperationsPrediction()

    def startValidation(self):

        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files for testing')

            # extracting values from training schema
            colNames, noofcolumns = self.raw_data.valuesfromSchema()

            # copy the file to the Prediction_Batch_files1 folder
            self.raw_data.savetoTestBatchFile()

            # copy the file to the Test_Raw_files_validated folder
            self.raw_data.putRawDatainTestFolder()

            # validate the number of columns is same as metioned in the schema
            self.raw_data.validateColumnLength(noofcolumns)

            # Check if any missing values present in any of the columns of the dataset
            self.raw_data.validateMissingValuesInWholeColumn()

            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            self.log_writer.log(self.file_object, "Starting Data Transformation!!")

            # replace blanks in the csv file as NULL
            self.dataTransform.replaceMissingWithNull_TestData()

            self.log_writer.log(self.file_object, "Data Transformation Completed!!!")

            self.log_writer.log(self.file_object,
                            "Creating Prediction_Database and tables on the basis of given schema!!!")

            # create database with given name, if present open the connection! Create table with columns given in schema
            self.dbOperation.connectCassandra()
            self.dbOperation.createTable('Prediction', colNames)

            self.log_writer.log(self.file_object, "Table creation Completed!!")
            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!!")

            # insert csv files in the table
            self.dbOperation.insertDataIntoTable('Prediction', colNames)

            self.log_writer.log(self.file_object, "Insertion in Table completed!!!")
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")

            # Delete the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataPredictionFolder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")

            # export data in table to csvfile
            self.dbOperation.selectingDatafromtableintocsv('Prediction')

        except Exception as e:
            raise Exception(e)


