from datetime import datetime
import os
import re
import csv
import json
import shutil
import pandas as pd
from os import listdir
from werkzeug.utils import secure_filename
from app_logging.logger import appLogger

class TestDataValidation:
    """
    This class will be used for handling all the validation done on the test raw data

    Written By: Anupam Hore
    Version: 1.0
    Revisions: None
    """
    def __init__(self,file):
        self.fileObj = file
        self.schema_path = 'schema_test.json'
        self.logger = appLogger()

    def savetoTestBatchFile(self):
        """
           Method Name: savetoTestBatchFile
           Description: This method saves the raw file to Prediction_Batch_files1 folder for further processing
           Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
           On Failure: Raise ValueError,KeyError,Exception

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        try:
            filename = secure_filename(self.fileObj.filename)
            self.fileObj.save(os.path.join('./Prediction_Batch_files1/', filename))

        except Exception as e:
            print('Exception: %s' % e)
            return Exception(e)

    def valuesfromSchema(self):
        """
           Method Name: valuesFromSchema
           Description: This method extracts all the relevant information from the pre-defined "Schema" file.
           Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
           On Failure: Raise ValueError,KeyError,Exception

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            column_names  = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            file = open("Prediction_Logs/valuesfromSchemaValidationLog.txt","a+")
            message = "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.logger.log(file,message)
            file.close()

        except ValueError:
            file = open("Prediction_Logs/valuefromSchemaValidationLog.txt", 'a+')
            self.logger.log(file,"ValueError:Value not found inside schema_training.json")
            file.close()
            raise ValueError

        except KeyError:
            file = open("Prediction_Logs/valuefromSchemaValidationLog.txt", 'a+')
            self.logger.log(file,"KeyError:Key value error incorrect key passed")
            file.close()
            raise KeyError

        except Exception as e:
            file = open("Prediction_Logs/valuefromSchemaValidationLog.txt", 'a+')
            self.logger.log(file,str(e))
            file.close()
            raise e

        return column_names, NumberofColumns


    def deleteExistingBadDataPredictionFolder(self):
        """
                   Method Name: deleteExistingBadDataPredictionFolder
                   Description: This method deletes the directory made to store the bad Data.
                   Output: None
                   On Failure: OSError

                   Written By: Anupam Hore
                   Version: 1.0
                   Revisions: None
        """
        try:
            path = 'Prediction_Raw_files_validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                file = open("Prediction_Logs/GeneralLog.txt",'a+')
                self.logger.log(file,"BadRaw Directory deleted before starting validation!!!")
                file.close()
        except OSError as s:
            file = open("Prediction_Logs/GeneralLog.txt",'a+')
            self.logger.lg(file, 'Error while deleting directory: %s' %s)
            file.close()
            raise OSError

    def deleteExistingGoodDataPredictionFolder(self):
        """
            Method Name: deleteExistingGoodDataPredictionFolder
            Description: This method deletes the directory made  to store the Good Data
                         after loading the data in the table. Once the good files are
                         loaded in the DB,deleting the directory ensures space optimization.
            Output: None
            On Failure: OSError

            Written By: Anupam Hore
            Version: 1.0
            Revisions: None
        """
        try:
            path = 'Prediction_Raw_files_validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                file = open("Prediction_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, 'GoodRaw directory deleted successfully!!!')
                file.close()
        except OSError as s:
            file = open("Prediction_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while deleting directory: %s" %s)
            file.close()
            raise OSError

    def createDirectoryForGoodBadRawData(self):
        """
           Method Name: createDirectoryForGoodBadRawData
           Description: This method creates directories to store the Good Data and Bad Data
                        after validating the test data.

           Output: None
           On Failure: OSError

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        try:
            path = os.path.join("Prediction_Raw_files_validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path  = os.path.join("Prediction_Raw_files_validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as s:
            file = open("Prediction_Logs/GeneralLog.txt",'a+')
            self.logger.log(file, "Error while creating directory %s" %s)
            file.close()
            raise OSError

    def putRawDatainTestFolder(self):
        """
           Method Name: putRawDatainTestFolder
           Description: This method will move the data to Prediction_Raw_files_validated/Good_Raw
                        from Prediction_Batch_files1 folder

           Output: None
           On Failure: OSError

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """

        #delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.deleteExistingBadDataPredictionFolder()
        self.deleteExistingGoodDataPredictionFolder()

        #create new directories
        self.createDirectoryForGoodBadRawData()
        try:
            f = open("Prediction_Logs/filecopiedLog.txt",'a+')

            filename = secure_filename(self.fileObj.filename)

            shutil.copy("Prediction_Batch_files1/" + filename, "Prediction_Raw_files_validated/Good_Raw")

            for file in listdir('Prediction_Raw_files_validated/Good_Raw/'):
                filename = secure_filename(self.fileObj.filename)
                self.logger.log(f, "Filename: " + str(filename))
                if filename.endswith('.csv'):
                    csv = pd.read_csv("Prediction_Raw_files_validated/Good_Raw/" + filename)
                    self.logger.log(f, "Read CSV File")
                else:
                    self.logger.log(f, "Read Excel File Started")
                    csv = pd.read_excel("Prediction_Raw_files_validated/Good_Raw/" + filename)
                    self.logger.log(f, "Read Excel File")

                splitArr = filename.split(".")
                self.logger.log(f, "splitArr[0]" + str(splitArr[0]))
                csv.to_csv("Prediction_Raw_files_validated/Good_Raw/" + splitArr[0] + ".csv",index=None, header=True)

                if os.path.exists("Prediction_Raw_files_validated/Good_Raw/" + file):
                    os.remove("Prediction_Raw_files_validated/Good_Raw/" + file)


            self.logger.log(f, "File copied to the Prediction_Raw_files_validated/Good_Raw folder")
            f.close()

        except Exception as e:
            f = open("Prediction_Logs/filecopiedLog.txt", 'a+')
            self.logger.log(f, "Error occurred while copy file %s" % e)
            f.close()
            raise e

    def validateColumnLength(self,noOfCols):
        """
            Method Name: validateColumnLength
            Description: This function validates the number of columns in the csv files.
                         It should be same as given in the schema file.
                         If not same, file is not suitable for processing and thus will be moved to Bad Raw Data folder.
                         If the column number matches, file is kept in Good Raw Data for processing.

            Output: None
            On Failure: Exception

            Written By: Anupam Hore
            Version: 1.0
            Revisions: None
        """
        try:
            f = open("Prediction_Logs/columnValidationLog.txt",'a+')
            self.logger.log(f,'Column Length Validation Started!!')
            for file in listdir('Prediction_Raw_files_validated/Good_Raw/'):
                filename = secure_filename(self.fileObj.filename)
                splitArr = filename.split(".")
                filename = splitArr[0] + ".csv"

                csv = pd.read_csv("Prediction_Raw_files_validated/Good_Raw/" + filename)

                if csv.shape[1] == noOfCols:
                    print('Same columns!!!!')
                else:
                    shutil.move("Prediction_Raw_files_validated/Good_Raw/" + file, "Prediction_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, 'Invalid column length for the file. File moved to Bad Raw Folder :: %s' %file)

            self.logger.log(f, "Column Length Validation Completed!!!")

        except OSError as s:
            f = open("Prediction_Logs/columnValidationLog.txt",'a+')
            self.logger.log(f, "Error Occurred while moving the file :: %s" %s)
            f.close()
            raise OSError

        except Exception as e:
            f = open("Prediction_Logs/columnValidationLog.txt",'a+')
            self.logger.log(f, "Error Occurred!!  %s" %e)
            f.close()
            raise e
        f.close()

    def validateMissingValuesInWholeColumn(self):
        """
            Method Name: validateMissingValuesInWholeColumn
            Description: This function validates if any column in the csv file has all values missing.
                         If all the values are missing, the file is not suitable for processing.
                         Such files are moved to bad raw data.
            Output: None
            On Failure: Exception

             Written By: Anupam Hore
             Version: 1.0
             Revisions: None
        """
        try:
            f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Missing Values Validation Started!!")



            for file in listdir('Prediction_Raw_files_validated/Good_Raw/'):
                filename = secure_filename(self.fileObj.filename)
                splitArr = filename.split(".")
                filename = splitArr[0] + ".csv"

                csv = pd.read_csv("Prediction_Raw_files_validated/Good_Raw/" + filename)

                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move("Prediction_Raw_files_validated/Good_Raw/" + file,
                                    "Prediction_Raw_files_validated/Bad_Raw")
                        self.logger.log(f, "Invalid Column for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
                if count == 0:
                    csv.to_csv("Prediction_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
                    self.logger.log(f, "No Missing Values found!!")
                self.logger.log(f, "Missing Values Validation Completed!!")
        except OSError:
            f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occurred while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occurred:: %s" % e)
            f.close()
            raise e
        f.close()