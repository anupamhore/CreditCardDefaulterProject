from datetime import datetime
import os
import re
import csv
import json
import shutil
import pandas as pd
from os import listdir
from app_logging.logger import appLogger

class DataValidation:
    """
    This class will be used for handling all the validation done on the raw data

    Written By: Anupam Hore
    Version: 1.0
    Revisions: None
    """

    def __init__(self,path):
        self.Batch_Directory = path
        self.schema_path = 'schema_training.json'
        self.logger = appLogger()

    def savetoTrainingBatchFile(self):
        """
           Method Name: savetoTrainingBatchFile
           Description: This method saves the raw file to Traning_Batch_Files folder for further processing
           Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
           On Failure: Raise ValueError,KeyError,Exception

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        try:
            shutil.copy(self.Batch_Directory, "Training_Batch_Files/")

        except Exception as e:
            print('Exception: %s' %e)
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

            file = open("Training_Logs/valuesfromSchemaValidationLog.txt","a+")
            message = "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.logger.log(file,message)
            file.close()

        except ValueError:
            file = open("Training_Logs/valuefromSchemaValidationLog.txt", 'a+')
            self.logger.log(file,"ValueError:Value not found inside schema_training.json")
            file.close()
            raise ValueError

        except KeyError:
            file = open("Training_Logs/valuefromSchemaValidationLog.txt", 'a+')
            self.logger.log(file,"KeyError:Key value error incorrect key passed")
            file.close()
            raise KeyError

        except Exception as e:
            file = open("Training_Logs/valuefromSchemaValidationLog.txt", 'a+')
            self.logger.log(file,str(e))
            file.close()
            raise e

        return column_names, NumberofColumns

    def deleteExistingBadDataTrainingFolder(self):
        """
                   Method Name: deleteExistingBadDataTrainingFolder
                   Description: This method deletes the directory made to store the bad Data.
                   Output: None
                   On Failure: OSError

                   Written By: Anupam Hore
                   Version: 1.0
                   Revisions: None
        """
        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                file = open("Training_Logs/GeneralLog.txt",'a+')
                self.logger.log(file,"BadRaw Directory deleted before starting validation!!!")
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt",'a+')
            self.logger.lg(file, 'Error while deleting directory: %s' %s)
            file.close()
            raise OSError

    def deleteExistingGoodDataTrainingFolder(self):
        """
            Method Name: deleteExistingGoodDataTrainingFolder
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
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file, 'GoodRaw directory deleted successfully!!!')
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while deleteing directory: %s" %s)
            file.close()
            raise OSError

    def createDirectoryForGoodBadRawData(self):
        """
           Method Name: createDirectoryForGoodBadRawData
           Description: This method creates directories to store the Good Data and Bad Data
                        after validating the training data.

           Output: None
           On Failure: OSError

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        try:
            path = os.path.join("Training_Raw_files_validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path  = os.path.join("Training_Raw_files_validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt",'a+')
            self.logger.log(file, "Error while creating directory %s" %s)
            file.close()
            raise OSError


    def putRawDatainTrainingFolder(self):
        """
           Method Name: putRawDatainTrainingFolder
           Description: This method will move the data to Training_Raw_files_validated/Good_Raw
                        from Training_Batch_Files folder

           Output: None
           On Failure: OSError

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """

        #delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.deleteExistingBadDataTrainingFolder()
        self.deleteExistingGoodDataTrainingFolder()

        #create new directories
        self.createDirectoryForGoodBadRawData()
        try:
            f = open("Training_Logs/filecopiedLog.txt",'a+')
            x = str(self.Batch_Directory)
            split = x.split("\\")
            ll = len(split) - 1
            filename = str(split[ll])
            shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Good_Raw")
            self.logger.log(f, "File copied to the Training_Raw_files_validated/Good_Raw folder")
            f.close()

        except Exception as e:
            f = open("Training_Logs/filecopiedLog.txt", 'a+')
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
            f = open("Training_Logs/columnValidationLog.txt",'a+')
            self.logger.log(f,'Column Length Validation Started!!')
            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                if csv.shape[1] == noOfCols:
                    print('Same columns!!!!')
                else:
                    shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, 'Invalid column length for the file. File moved to Bad Raw Folder :: %s' %file)

            self.logger.log(f, "Column Length Validation Completed!!!")

        except OSError as s:
            f = open("Training_Logs/columnValidationLog.txt",'a+')
            self.logger.log(f, "Error Occurred while moving the file :: %s" %s)
            f.close()
            raise OSError

        except Exception as e:
            f = open("Training_Logs/columnValidationLog.txt",'a+')
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
            f = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Missing Values Validation Started!!")



            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count += 1
                        shutil.move("Training_Raw_files_validated/Good_Raw/" + file,
                                    "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(f, "Invalid Column for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
                if count == 0:
                    csv.rename(columns={"default.payment.next.month": "TARGET"}, inplace=True)
                    csv.to_csv("Training_Raw_files_validated/Good_Raw/" + file, index=None, header=True)
                    self.logger.log(f, "No Missing Values found!!")
                self.logger.log(f, "Missing Values Validation Completed!!")
        except OSError:
            f = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occurred while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Training_Logs/missingValuesInColumn.txt", 'a+')
            self.logger.log(f, "Error Occurred:: %s" % e)
            f.close()
            raise e
        f.close()
