import pandas as pd
from app_logging.logger import appLogger
from data_preprocessing.preprocessing import Preprocessor
from data_preprocessing.clustering import KMeansClustering
from data_transformation.datatransformation import DataTransform
from feature_selection.featureSelection import FeatureSelection
import numpy as np
from sklearn.model_selection import train_test_split
from file_ops import file_methods
from best_model_finder.tuner import Model_Finder

class TrainModel:

    def __init__(self):
        self.logger = appLogger()

        pass

    def modelTraining(self):
        """
        Method Name: modelTraining
        Description:This class trains the dataset after doing all the preprocessing
        Output: None
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
        self.logger.log(self.file_object, 'Start of Training')

        try:
            #get the data
            self.df = pd.read_csv('Training_FileFromDB/InputFile.csv')

            #initiate the preprocessor class
            preprocessor = Preprocessor(self.file_object,self.logger)

            X, Y = preprocessor.separate_label_features(self.df, 'target')

            #find if any missing value present in the data
            isnullpresent, missingValueColumns = preprocessor.is_null_present(X)

            #impute missing values
            if isnullpresent:
                X = preprocessor.impute_missingValues(X,missingValueColumns)

            #initial columns to drop
            cols_to_drop =['Unnamed: 0','id']
            X = preprocessor.dropVariables(X,cols_to_drop,1,True)

            #outlier treatment
            cols_needing_outlierTreatment = ['age','bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6',
                                             'education','limit_bal','pay_0','pay_2','pay_3','pay_4','pay_5','pay_6',
                                             'pay_amt1','pay_amt2','pay_amt3','pay_amt4','pay_amt5','pay_amt6']
            distributionTypes =['Skewed','Highly Skewed','Highly Skewed','Highly Skewed','Highly Skewed','Highly Skewed','Highly Skewed',
                                'Skewed','Skewed','Gaussian','Gaussian','Gaussian','Gaussian','Gaussian','Gaussian',
                                'Highly Skewed','Highly Skewed','Highly Skewed','Highly Skewed','Highly Skewed','Highly Skewed']
            isLengthSame, X = preprocessor.outlier_treatment(X, cols_needing_outlierTreatment,distributionTypes)


            if isLengthSame:

                #Separate the categorical and continous variables from the input variables for processing
                cagtegoricalVars, numericalVars = preprocessor.separate_cat_num(X)

                self.logger.log(self.file_object, "Categorical Variables: {}".format(cagtegoricalVars))
                self.logger.log(self.file_object, "Numerical Variables: {}".format(numericalVars))

                #Statistically find out the lease significant variables. This variables can be dropped during feature selection
                least_significantVars = preprocessor.find_statistical_least_significant_variable(X,cagtegoricalVars,numericalVars,Y)

                self.logger.log(self.file_object, "Least Significant Variables: {}".format(least_significantVars))

                #check for any transformation is required or not
                data_trans_obj = DataTransform(self.logger)
                vars_needed_transformation = data_trans_obj.checkforTransformation(X, numericalVars, Y)

                self.logger.log(self.file_object, "Variables for transformation: {}".format(vars_needed_transformation))


                #check for multi-collinearity
                high_collinear_vars = preprocessor.checkforMultiCollinearity(X,0.7)
                self.logger.log(self.file_object, "High Multi-Collinear Variables: {}".format(high_collinear_vars))


                #perform feature selection
                featureSelection = FeatureSelection(self.logger)

                #find out features which has low Somer's D. High SomersD for continous variable is good
                low_somersD_vars = featureSelection.findlowSomersD_vars(X, Y)
                self.logger.log(self.file_object, "Low Somer's D Variables: {}".format(low_somersD_vars))

                #find features whose VIF values are less than 10. Those features will be good for the model
                features_with_low_vif = featureSelection.findVIF_Factor(X)
                self.logger.log(self.file_object, "Low VIF Variables: {}".format(features_with_low_vif))

                #find constant features
                const_features = featureSelection.findConstantFeatures(X)
                # we will drop the constant features
                if len(const_features) > 0:
                    X.drop(const_features,axis=1,inplace=True)


                #Feature Selection will be performed based on the following variables
                # 1. least_significantVars
                # 2. high_collinear_vars
                # 3. low_somersD_vars
                # 4. features_with_low_vif
                # After finalizing the variables we will also consider
                # "vars_needed_transformation" to check amongst the variables got from the above filtering,
                # which variable will require transform
                # Finally we will use those variables for model training

                #find out final features for model
                X_features = featureSelection.findFinalFeatures(least_significantVars, high_collinear_vars,
                                                                low_somersD_vars, features_with_low_vif)

                self.logger.log(self.file_object, "Final Features before transformation: {}".format(X_features))

                #replace the final features in the dataset from the feature selection process
                X = featureSelection.updateDataSet(X,X_features)

                # do the transformation for the variables
                X = data_trans_obj.transformVariables(X, X_features, vars_needed_transformation)

                self.logger.log(self.file_object, "Final Features after transformation: {}".format(X.columns))

                # find if any missing value present in the data
                isnullpresent1, missingValueColumns1 = preprocessor.is_null_present(X)

                # impute missing values
                if isnullpresent1:
                    X = preprocessor.impute_missingValues(X, missingValueColumns1)

                """ Applying the clustering approach"""

                # standard scaler
                data_scaled_X1 = preprocessor.scaleData(X)

                kmeans = KMeansClustering(self.logger)
                number_of_clusters = kmeans.elbow_plot(data_scaled_X1)

                self.logger.log(self.file_object, "Total number of clusters formed: {}".format(number_of_clusters))

                # Divide the data into clusters
                data_scaled_X2 = kmeans.create_clusters(data_scaled_X1, number_of_clusters)
                data_scaled_X2['Labels'] = Y

                list_of_unique_clusters = data_scaled_X2['Cluster'].unique()

                for cluster in list_of_unique_clusters:
                    cluster_data = data_scaled_X2[data_scaled_X2['Cluster'] == cluster]

                    # Prepare the dependant and independant variables
                    cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                    cluster_label = cluster_data['Labels']

                    # split the data
                    X_train, X_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.30,
                                                                        random_state=100)

                    model_finder = Model_Finder(self.logger)

                    # getting the best model for each of the clusters
                    best_model_name, best_model = model_finder.get_best_model(X_train, y_train, X_test, y_test)

                    # save the best model to the directory
                    file_op = file_methods.File_Operation()
                    save_model = file_op.save_model(best_model, best_model_name + str(cluster))

                    if save_model == 'success':
                        self.logger.log(self.file_object,"Model:{} saved for Cluster:{}".format(best_model_name, str(cluster)))


            else:
                self.logger.log(self.file_object,'Cannot Proceed because of length mismatch for distributions and cols list')
            self.file_object.close()

        except Exception as e:
            self.logger.log(self.file_object, 'Model training unsuccesfull because: %s'%Exception(e))
            self.file_object.close()
            raise Exception(e)

