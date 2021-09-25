import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from app_logging.logger import appLogger
from cassandra import metadata
import os
import csv
import pandas as pd
from pathlib import  Path

class DBOperationsPrediction:
    """
    This class shall be used for database CRUD operations

    Written By: Anupam Hore
    Version: 1.0
    Revisions: None
    """
    
    def __init__(self):
        self.cloud_config = {'secure_connect_bundle': "cassandraconnection\\secure-connect-test.zip"}
        self.auth_provider = PlainTextAuthProvider('InGidfaCfiNMUlbTqiUBJoKd',
                                                    'lfsErHI40dg3xMF9HP4Jac_8zx8jcsEwLHt1hc7LOQ7NGnPcZkiXkADu.q_Mscoeg8JoYWbu9zs3YZ,cKWE6rJHK27MU9FnJM1,BSvYvvJm1pC18hpa,kmnQfEeSBY6Z')
        self.cluster = Cluster(cloud=self.cloud_config, auth_provider=self.auth_provider)
        self.goodDataPath = "Prediction_Raw_files_validated/Good_Raw"
        self.logger = appLogger()

    def connectCassandra(self):
        """
        Method Name: connectCassandra
        Description: This method connects to the cloud Cassandra database

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        try:
            f = open("Prediction_Logs/dbOperations.txt", 'a+')
            self.session = self.cluster.connect()

            row = self.session.execute("select release_version from system.local").one()
            if row:
                self.logger.log(f, "Database Connection Established %s" %row[0])
            else:
                self.logger.log(f,"Database connection Failed!!!")
            f.close()
        except Exception as e:
            f = open("Prediction_Logs/dbOperations.txt", 'a+')
            self.logger.log(f, "Database connection Failed with %s" %Exception(e))
            f.close()
            self.cluster.shutdown()
            raise Exception(e)

    def createTable(self,tableName,column_names):
        """
        Method Name: createTable
        Description: This method will create the table in the cloud Cassandra database with the colNames as entries
        Parameters: tableName, colNames
        Output: None
        On Failure: Raise Exception
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        try:

            f = open("Prediction_Logs/dbOperations.txt", 'a+')
            self.logger.log(f, "Columns:{}".format(column_names))
            keyspaces_list = self.cluster.metadata.keyspaces
            for keyspace in keyspaces_list:
                if keyspace == 'anupam':
                    self.logger.log(f, "anupam keyspace is located in Cassandra DB!!!")
                    keyspace_name = self.cluster.metadata.keyspaces[keyspace]
                    tables = keyspace_name.tables
                    self.session.execute('DROP TABLE IF EXISTS anupam.{}'.format(tableName))
                    for key in column_names.keys():
                        self.logger.log(f, "{} column to create".format(key))
                        type = column_names[key]
                        try:
                            self.session.execute('ALTER TABLE anupam.{tablename} ADD {column_name} {dataType}'.format(tablename=tableName, column_name=key, dataType=type))
                            self.logger.log(f, "{} column created in Cassandra keyspace!!!".format(key))
                        except:
                            self.session.execute('CREATE TABLE anupam.{tablename} ({column_name} {dataType} PRIMARY KEY)'.format(tablename=tableName, column_name=key, dataType=type))
                            self.logger.log(f, "Table and first column in Cassandra keyspace!!!")

                    break
            self.logger.log(f, "Table Creation completed in Cassandra keyspace!!!")
            f.close()


        except Exception as e:
            f = open("Prediction_Logs/dbOperations.txt", 'a+')
            self.logger.log(f, "Table not created with Exception %s" %Exception(e))
            f.close()
            self.cluster.shutdown()
            raise Exception(e)

    def insertDataIntoTable(self,tableName,column_names):
        """
           Method Name: insertDataIntoTable
           Description: This method inserts the Good data files from the Good_Raw folder into the
                        above created table.
           Output: None
           On Failure: Raise Exception

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """

        columns = []
        for key in column_names.keys():
            columns.append(key)
        colStr = ''
        formatter = ''
        count = 0
        for col in columns:
            if count < len(columns) - 1:
                colStr = colStr + str(col) + " ,"
                formatter = formatter + "%s" + ","
            else:
                colStr = colStr + str(col)
                formatter = formatter + "%s"

            count = count + 1

        try:
            log_file = open("Prediction_Logs/dbOperations.txt", 'a+')
            onlyfiles = [f for f in os.listdir(self.goodDataPath)]
            self.logger.log(log_file, "Insertion Started!!!")
            for file in onlyfiles:
                try:
                    with open(self.goodDataPath + '/' + file, "r") as f:
                        next(f)
                        data = csv.reader(f, delimiter="\n")
                        for line in data:
                            lis = line[0].split(',')
                            lis[0] = int(lis[0])

                            try:

                                #print('Started Insersion')
                                #print('insert into anupam.{tablename}({col}) values({formatter});'.format(tablename=tableName,col=colStr,formatter=formatter))
                                self.session.execute('insert into anupam.{tablename}({col}) values({formatter});'.format(tablename=tableName,col=colStr,formatter=formatter),lis)
                            except Exception as e:
                                self.logger.log(log_file, "Insertion Failure at Cassandra Cloud: %s" % Exception(e))
                                #log_file.close()
                                raise Exception(e)

                except Exception as e:
                    self.logger.log(log_file, "File Operation Failed: %s" %Exception(e))
                    #log_file.close()
                    raise Exception(e)
            self.logger.log(log_file, "Insertion Completed!!!")
            log_file.close()

        except Exception as e:
            log_file = open("Prediction_Logs/dbOperations.txt", 'a+')
            self.logger.log(log_file, "DB Insertion Failure: %s" % Exception(e))
            log_file.close()
            self.cluster.shutdown()
            raise Exception(e)

    def selectingDatafromtableintocsv(self,tableName):
        """

        Method Name: selectingDatafromtableintocsv
        Description: This method exports the data from Cassandra DB as a CSV file. in a  location.
                       created at (Prediction_FileFromDB)
        Output: None
        On Failure: Raise Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """

        self.fileFromDb = 'Prediction_FileFromDB/'
        self.fileName = 'InputFile.csv'
        log_file = open("Prediction_Logs/ExportToCsv.txt", 'a+')
        try:
            self.logger.log(log_file, "File exported started!!!")

            # Only when doing for development
            # This code will be commented and the else part will work.
            input_file = Path(self.fileFromDb + self.fileName)
            if input_file.is_file():
                # file exists
                self.logger.log(log_file, "File already existed!!!")
                # delete the old file
                os.remove(input_file)
            # else:
            rows = self.session.execute('select * from anupam.{tablename}'.format(tablename=tableName))
            df = pd.DataFrame(rows.all())
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.to_csv(self.fileFromDb + self.fileName)
            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()
            self.cluster.shutdown()

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" %e)
            log_file.close()
            self.cluster.shutdown()
            raise Exception(e)

