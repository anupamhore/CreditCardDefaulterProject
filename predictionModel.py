from app_logging.logger import appLogger
import pandas as pd
from data_preprocessing.preprocessing import Preprocessor
from feature_selection.featureSelection import FeatureSelection
from file_ops import file_methods
from sys import platform
import os
import shutil
from datetime import datetime

class PredictModel:
    def __init__(self):
        self.logger = appLogger()

    def predictTestData(self):
        """
        Method Name: predictTestData
        Description:This class predicts the dataset after doing all the preprocessing
        Output: None
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.file_object = open("Prediction_Logs/Prediction.txt", 'a+')
        self.logger.log(self.file_object, 'Start of Prediction!!!')

        try:
            #get the data
            self.df = pd.read_csv('Prediction_FileFromDB/InputFile.csv')

            # initiate the preprocessor class
            preprocessor = Preprocessor(self.file_object, self.logger)

            # find if any missing value present in the data
            isnullpresent, missingValueColumns = preprocessor.is_null_present(self.df)

            # impute missing values
            if isnullpresent:
                self.df = preprocessor.impute_missingValues(self.df, missingValueColumns)

            #initial columns to drop
            # cols_to_drop =['Unnamed: 0']
            cols_to_drop = 'Unnamed: 0'
            X = preprocessor.dropVariablesTest(self.df,cols_to_drop,1,True)

            # perform feature selection
            featureSelection = FeatureSelection(self.logger)

            #only keep this variables and drop the rest
            X_features = ['id','limit_bal', 'pay_0', 'pay_2', 'pay_amt1', 'pay_amt2', 'pay_amt3',
                   'pay_amt4', 'pay_amt5', 'pay_amt6']

            X = featureSelection.updateDataSet(self.df, X_features)

            # standard scaler
            # X = preprocessor.scaleDataTransform(X)

            # load the KMeans model to the directory
            file_op = file_methods.File_Operation()
            kmeans = file_op.load_model('KMeans')

            clusters = kmeans.predict(X.drop(['id'],axis=1))  # drops the first column for cluster prediction
            X['clusters'] = clusters
            clusters = X['clusters'].unique()

            for i in clusters:

                cluster_data = X[X['clusters'] == i]
                customer_id_list = list(cluster_data['id'])

                cluster_data = cluster_data.drop(['clusters','id'], axis = 1)
                model_name = file_op.find_correct_model_file(i)
                model = file_op.load_model(model_name)
                finalDF = pd.DataFrame()
                if model_name.find("Extreme") != -1:
                    model1 = file_op.load_model('Naive Bayes0')
                    result = list(model1.predict(cluster_data))
                    result = pd.DataFrame(list(zip(customer_id_list,result)),columns=['ID','Prediction'])
                    finalDF = pd.concat([finalDF,result],axis=1)

                else:
                    result = list(model.predict(cluster_data))
                    result = pd.DataFrame(list(zip(customer_id_list,result)),columns=['ID','Prediction'])

                    finalDF = pd.concat([finalDF, result], axis=1)

            finalDF.to_csv("Prediction_Output_File/Predictions.csv", header=True)

            # desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            # path = os.path.join(desktop, "Predictions.csv")
            # self.logger.log(self.file_object, 'End Path: %s'%path)
            #
            # shutil.copy("Prediction_Output_File/Predictions.csv", path)


            self.logger.log(self.file_object, 'End of Prediction')
            endTime = datetime.now().replace(microsecond=0)

            return endTime

        except Exception as e:
            self.logger.log(self.file_object, 'Error occurred while running the prediction!! Error:: %s' % e)
            raise e
        # , finalDF.head(30).to_json(orient="records")


















