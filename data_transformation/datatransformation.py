import pandas as pd
import numpy as np
from data_preprocessing.preprocessing import Preprocessor
import statsmodels.formula.api as sm

class DataTransform:
    """
        This class shall  be used to transform the data and check which continous variables need transformation

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
    """
    def __init__(self, logger):
        self.logger = logger
        self.file_object = open("Training_Logs/DataTransformation.txt", 'a+')

    def checkforTransformation(self, df, numericalVars, target):
        """
        Method Name: checkforTransformation
        Description: This function checks which continous variables needed transformation and
                     after transformation whether the variables have better explainatory power to define
                     the relationship with the target.
                     The variables are transformed into
                     (i)Squared of the variable,
                     (ii)Square Root of the variable,
                     (iii)Log of the variable
                     These three transformation along with the original variable are then compared by calculating
                     the loglikelihood,and the var with greater loglikelihood is considered.
        Parameter: df ( dataframe) , numericalVars, target variable
        Output: list of transformed variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """


        try:
            self.logger.log(self.file_object, "checkforTransformation Started!!!")
            df_transform = df[numericalVars].copy()
            squared = (df_transform ** 2).add_suffix("_squared")
            square_root = (df_transform ** 0.5).add_suffix("_sqrt")
            natural_log = np.log(df_transform + 1).add_suffix("_ln")
            df_transform = pd.concat([df_transform, squared, square_root, natural_log], axis=1)
            df_transform['target'] = target

            preprocessor = Preprocessor(self.file_object, self.logger)
            X = df_transform.drop('target',axis=1)
            Y = df_transform['target']

            #find if any missing value present in the data
            isnullpresent, missingValueColumns = preprocessor.is_null_present(X)

            if isnullpresent:
                X = preprocessor.impute_infs(X,missingValueColumns)

            #find one more time
            isnullpresent1, missingValueColumns1 = preprocessor.is_null_present(X)

            if isnullpresent1:
                X = preprocessor.impute_missingValues(X,missingValueColumns1)


            #find out the log-likelihood of the variables using the statsmodel
            df = pd.concat([X, Y], axis=1)
            vars_list = self.calculate_loglikelihood(df)

            self.logger.log(self.file_object, "checkforTransformation Completed!!!")
            #self.file_object.close()
            return vars_list


        except Exception as e:
            self.logger.log(self.file_object, "Transformation Failed: %s"%Exception(e))
            #self.file_object.close()
            raise Exception(e)

    def calculate_loglikelihood(self, df_transform):
        """
        Method Name: calculate_loglikelihood
        Description: This function calculate the loglikelihood of all the numerical
                     variables and find out which variable after its transformation
                     is having greater loglikelihood. The variable which is transformed
                     is checked and compared to get the best loglikelihood value.
                     That transformated variable is considered and returned.

        Parameter: df_transform ( dataframe)
        Output: list of transformed variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """

        self.logger.log(self.file_object, "calculate_loglikelihood Started!!!")
        try:
            lldf = pd.DataFrame()
            ft = ''
            for feature in df_transform.columns.difference(['target']):
                ft = feature
                logreg = sm.logit(formula=str('target ~') + str(feature), data=df_transform)
                result = logreg.fit()
                tmp = pd.DataFrame([feature, result.llf]).T
                tmp.columns = ['Feature', 'Log-Likelihood Full']
                lldf = pd.concat([lldf, tmp], axis=0)

            arr = lldf.to_numpy() #convert the dataframme to np array
            arr1 = []
            for list_ in arr:
                name = list_[0] # Feature Name
                namealias = list_[0]
                value = list_[1] # LogLikelihood Value
                isPresent = False
                namealias11 = ''
                if name.find("ln") >= 0 or name.find("sqrt") >= 0 or name.find('squared') >= 0:
                    split1 = name.split("_")
                    namealias1 = ''
                    count = 0
                    for i in range(len(split1) - 1):
                        if count == 0 and (len(split1) - 1) > 1:
                            namealias1 = namealias1 + split1[i] + "_"
                        else:
                            namealias1 = namealias1 + split1[i]
                        count = count + 1
                    namealias11 = namealias1
                else:
                    namealias11 = name

                for item in arr1:
                    if item.find(namealias11) >= 0:
                        isPresent = True
                        break

                counter11 = 0
                for item in arr:
                    if item[0] == name:
                        break;
                    else:
                        counter11 = counter11 + 1

                if isPresent == False:
                    for list1_ in arr[counter11:]:  # Problem is here for next item
                        if list1_[0].find(name) != -1:
                            tmpVal = list1_[1]
                            if tmpVal > value:
                                value = tmpVal
                                namealias = list1_[0]

                        else:
                            arr1.append(namealias)
                            break

            self.logger.log(self.file_object, "calculate_loglikelihood Completed!!!")
            #self.file_object.close()
            return arr1


        except Exception as e:
            self.logger.log(self.file_object,"Error in calculating loglikelihood: %s"%Exception(e))
            #self.file_object.close()
            raise Exception(e)


    def transformVariables(self,X,X_features,vars_needed_transformation):
        """
        Method Name: transformVariables
        Description: This function first compares the original features(X_features)
                     and the transformed features(vars_needed_transformation) and find out
                     the feature in the original form which needs transformation.
                     Then it transformed that variable and set it to the dataframe X and returns it


        Parameter: X ( dataframe), X_features(independant features), vars_needed_transformation(name of the transformed features)
        Output: Transformed DataFrame
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """

        self.logger.log(self.file_object, "transformVariables Started!!!")
        try:
            """
            Logic: Iterate through the variables in vars_needed_transformation
                   Split them based on "_"
                   item_ will be the last item of splitarr which can take either values:
                       "squared", "ln", "sqrt", or some part of original variable
                   The if condition checks whether the "item_" contains any of the above values except for some part of original variable
                   If it contains some part of original variable -> it means no transformation is required
                   Otherwise:
                             The if conditions runs and we first of all assemble the variable name without the "squared", "ln", "sqrt" on it
                             Then we run a loop through the X_features(which are the original features)
                             And we try to map any of the features with the one we got in the if condition
                             If it matches, then we take the transformed variables type which may be either of them ( "squared", "ln", "sqrt")
                             Once we find, we transform the original variable values in X dataset to that
            """
            for item in vars_needed_transformation:
                splitarr = item.split("_")
                item_ = splitarr[len(splitarr) - 1]
                if item_.find("squared") >= 0 or item_.find("ln") >= 0 or item_.find("sqrt") >= 0:

                    item1_ = ''
                    count = 0
                    for i in splitarr:
                        if count < len(splitarr) - 1:
                            if count < len(splitarr) - 2:
                                item1_ = item1_ + i + "_"
                            else:
                                item1_ = item1_ + i

                        count = count + 1
                    #print(item1_)

                    for itemX in X_features:

                        if itemX == item1_:
                            # transform the variable
                            if item_.find("squared") >= 0:
                                X[itemX] = X[itemX] ** 2
                            elif item_.find("sqrt") >= 0:
                                X[itemX] = X[itemX] ** 0.5
                            else:
                                X[itemX] = np.log(X[itemX] + 1)
                            break


            self.logger.log(self.file_object, "transformVariables Completed!!!")
            self.file_object.close()
            return X

        except Exception as e:
            self.logger.log(self.file_object, "transformVariables failed with error: %s"%Exception(e))
            self.file_object.close()
            raise Exception(e)



