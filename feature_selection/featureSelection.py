import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold

class FeatureSelection:
    """
        This class shall  be used to select the best possible features for model training

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
    """
    def __init__(self, logger):
        self.logger = logger
        self.file_object = open("Training_Logs/FeatureSelection.txt", 'a+')


    def findlowSomersD_vars(self, X, Y):
        """
        Method Name: findlowSomersD_vars
        Description: This function checks which continous variable has the low Somer's D.

        Parameter: X ( independant variables), Y (target variable)
        Output: list of variables with low Somer's D
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """

        self.logger.log(self.file_object,"findlowSomersD_vars Started!!!")
        try:
            somersD_df = pd.DataFrame()
            df_transform1 = pd.concat([X,Y],axis=1)
            for feature in df_transform1.columns.difference(['target']):
                logreg = sm.logit(formula='target ~' + str(feature), data=df_transform1)
                result = logreg.fit()
                y_score = pd.DataFrame(result.predict())
                y_score.columns = ['Score']
                somers_d = 2 * metrics.roc_auc_score(df_transform1['target'], y_score) - 1
                tmp = pd.DataFrame([feature, somers_d]).T
                tmp.columns = ['Feature', 'SomersD']
                somersD_df = pd.concat([somersD_df, tmp], axis=0)

            arr = somersD_df.to_numpy()
            low_somersD_var = []
            for list_ in arr:
                if list_[1] < 0.1:
                    low_somersD_var.append(list_[0])


            self.logger.log(self.file_object, "findlowSomersD_vars Completed!!!")
            #self.file_object.close()
            return low_somersD_var

        except Exception as e:
            self.logger.log(self.file_object, "Finding Somer'sD Error: %s"%Exception(e))
            #self.file_object.close()
            raise Exception(e)

    def findVIF_Factor(self, X):
        """
        Method Name: findVIF_Factor
        Description: This function finds out the variables with low VIF factors. Those variables
                     can be considered for model training

        Parameter: X ( independant variables)
        Output: list of variables with low VIF
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"findVIF_Factor method Started!!!!")
        try:
            scaler = StandardScaler()
            arr = scaler.fit_transform(X)
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
            vif['Feature'] = X.columns
            series = vif[vif['VIF'] < 10]
            arr = list(series.Feature)

            self.logger.log(self.file_object, "findVIF_Factor method Completed!!!!")
            #self.file_object.close()
            return arr

        except Exception as e:
            self.logger.log(self.file_object, "findVIF_Factor error: %s"%Exception(e))
            #self.file_object.close()
            raise Exception(e)

    def findFinalFeatures(self, least_significantVars, high_collinear_vars, low_somersD_vars, features_with_low_vif):
        """
        Method Name: findFinalFeatures
        Description: This function filters out the variables from the given arguments and gives the best
                     features for model training

        Parameter: high_collinear_vars, low_somersD_vars, features_with_low_vif
        Output: list of best variables for model training
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"findFinalFeatures Started!!!")

        try:

            #First combine the low performing variables
            combined_list = least_significantVars + high_collinear_vars + low_somersD_vars
            combined_list_set = set(combined_list)
            combined_list = list(combined_list_set)

            #Second remove them from the high performing variables in features_with_low_vif
            final_features = [feature for feature in features_with_low_vif if feature not in combined_list]

            self.logger.log(self.file_object, "findFinalFeatures Completed!!!")
            #self.file_object.close()
            return final_features

        except Exception as e:
            self.logger.log(self.file_object, "findFinalFeatures Failed error: %s"%Exception(e))
            #self.file_object.close()
            raise Exception(e)

    def updateDataSet(self,X,X_features):
        """
        Method Name: updateDataSet
        Description: This function updates the dataset with the features selected after feature selection process

        Parameter: X(dataset), X_features(features important for the model)
        Output: updated dataset
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "updateDataSet Started!!!")
        try:
            drop_list = [feature for feature in X.columns if feature not in X_features]
            X.drop(drop_list,axis=1,inplace=True)

            self.logger.log(self.file_object, "updateDataSet Completed!!!")
           # self.file_object.close()
            return X

        except Exception as e:
            self.logger.log(self.file_object, "Update Dataset Failed: %s"%Exception(e))
           # self.file_object.close()
            raise Exception(e)

    def findConstantFeatures(self,data):
        """
        Method Name: findConstantFeatures
        Description: This function helps to find out the variables with constant values.
                     Sometime, the data might be constant values. These variables are not
                     good for model traning

        Parameter: X(dataset), X_features(features important for the model)
        Output: list of const columns
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "findConstantFeatures Started!!!")
        try:
            var_threshold = VarianceThreshold(threshold=0)
            var_threshold.fit(data)
            #var_threshold.get_support() returns True,False for each variable
            const_columns = [column for column in data.columns if column not in data.columns[var_threshold.get_support()]]
            self.logger.log(self.file_object, "findConstantFeatures Completed!!!")
            return const_columns

        except Exception as e:
            self.logger.log(self.file_object, "findConstantFeatures Fail: %s"%Exception(e))
            raise Exception(e)


