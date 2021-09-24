import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None

    """
    def __init__(self, file_object,logger):
        self.logger = logger
        self.file_object = file_object

    def separate_label_features(self, df, labelName):
        """
        Method Name: separate_label_features
        Description: This function separates the independant and dependant variables
        Parameter: df,labelName ( Dataframe and the target Variable name)
        Output: dependant variables, independant variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        try:
            self.logger.log(self.file_object, "separate_label_features method Started")
            X = df.drop(labelName, axis=1)
            Y = df[labelName]
            self.logger.log(self.file_object, "separate_label_features method Completed")
            return X, Y
        except Exception as e:
            self.logger.log(self.file_object, "Error in separating the target and independant variables %s"%Exception(e))
            raise Exception(e)

    def is_null_present(self, df):
        """
        Method Name: is_null_present
        Description: This function validates if any column in the dataframe has any missing values
        Parameter: df(the dataframe)
        Output: True or False( if missing values present in the dataset) & the columns which have missing values
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "Missing Value Search Started")
        null_present = False
        cols = df.columns
        columns_with_missing_values = []
        try:

            null_counts = df.isnull().sum()
            for i in range(len(null_counts)):
                if null_counts[i] > 0:
                    null_present = True
                    columns_with_missing_values.append(cols[i])

            if null_present:
                self.logger.log(self.file_object,"Missing Value Columns are %s"%columns_with_missing_values)
            self.logger.log(self.file_object,"Missing Values Search Completed")
            return null_present, columns_with_missing_values

        except Exception as e:
            self.logger.log(self.file_object,"Missing Value Search Error: %s" %Exception(e))
            raise Exception(e)

    def impute_missingValues(self, df, cols):
        """
        Method Name: impute_missingValues
        Description: This function imputes the missing values based on the type.
                     For continous numerical variable we use median()
                     For discrete numerical variable we use mode()
                     For categorical variable we use mode()
        Parameter: df(the dataframe), cosl ( the missing value columns list)
        Output: Dataset with missing values imputed
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "impute_missingValues Started!!!")
        try:
            for col in cols:
                if df[col].dtypes == 'object':
                    df[col] = df[col].fillna(df[col].mode())
                else:
                    discreteCol = pd.Categorical(df[col])
                    if len(discreteCol.categories) > 19:
                        #continous data
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        # discrete variable
                        df[col] = df[col].fillna(df[col].mode())

            self.logger.log(self.file_object, "impute_missingValues Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "Error imputing missing values %s" %Exception(e))
            raise Exception(e)

    def impute_infs(self,df,cols):
        """
        Method Name: impute_infs
        Description: This function imputes the infs values present in the dataset
        Parameter: df(the dataframe), cosl ( the missing value(infs) columns list)
        Output: Dataset with missing values imputed
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "impute_infs Started!!!")
        try:
            for col in cols:
                if np.isinf(df[col]).sum() > 0:
                    x = df[col]
                    x[np.isneginf(x)] = x.median()

            self.logger.log(self.file_object, "impute_infs Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "Infs imputation error: %s"%Exception(e))
            raise Exception(e)


    def dropVariables(self,df,cols,axis,inplace):
        """
        Method Name: dropVariables
        Description: This function drops the respective cols from the dataframe
        Parameter: df(the dataframe), cols ( the columns), axis (0,1), inplace(True,False)
        Output: Dataset with dropped variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None

        """
        self.logger.log(self.file_object, "dropVariables Started!!!")
        try:
            df.drop(cols,axis=axis,inplace=inplace)
            self.logger.log(self.file_object, "dropVariables Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "Error in dropping variables due to %s"%Exception(e))
            raise Exception(e)


    def outlier_treatment(self,df, cols,distType):
        """
        Method Name: outlier_treatment
        Description: This function is responsible for the outlier treatment
        Parameter: df(the dataframe), cols ( the columns), distType(list of variable distribution types)
                   Depending on the distribution, we will treat the outliers for the variables
        Output: Dataset with outlier imputed
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "outlier_treatment Started!!!")
        isLengthEqual = True
        try:
            if len(cols) == len(distType):
                count = 0
                for col in cols:
                    distributionType = distType[count]

                    if distributionType == "Gaussian":
                        upper_bound = df[col].mean() + 3 * df[col].std()
                        lower_bound = df[col].mean() - 3 * df[col].std()
                        df[col].clip(upper=upper_bound, inplace=True)
                        df[col].clip(lower=lower_bound, inplace=True)

                    elif distributionType == "Skewed":
                        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
                        upper_bound = df[col].quantile(0.75) + 1.5 * IQR
                        lower_bound = df[col].quantile(0.25) - 1.5 * IQR
                        df[col].clip(upper=upper_bound, inplace=True)
                        df[col].clip(lower=lower_bound, inplace=True)

                    else:
                        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
                        upper_bound = df[col].quantile(0.75) + 3 * IQR
                        lower_bound = df[col].quantile(0.25) - 3 * IQR
                        df[col].clip(upper=upper_bound, inplace=True)
                        df[col].clip(lower=lower_bound, inplace=True)

                    count = count + 1
                self.logger.log(self.file_object, "outlier_treatment Completed!!!")
                return isLengthEqual, df

            else:
                isLengthEqual = False
                self.logger.log(self.file_object, "Length of columns and length of distributions mismatch!!!")
                return isLengthEqual, df

        except Exception as e:
            self.logger.log(self.file_object, "Outlier Imputation Error: %s"%Exception(e))
            raise Exception(e)

    def separate_cat_num(self,df):
        """
        Method Name: separate_cat_num
        Description: This function separates the categorical variables and the numerical variable
        Parameter: df(the dataframe)
        Output: categorical vars, numerical vars
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "separate_cat_num Started!!!")
        try:
            categorical_features = []
            numerical_features = []
            features = df.columns
            for feature in features:
                if df[feature].dtypes == 'object':
                    categorical_features.append(feature)
                else:
                    #check if the feature is a discrete variable or not
                    discreteVar = pd.Categorical(df[feature])
                    if len(discreteVar.categories) > 19:
                        #that means its numerical
                        numerical_features.append(feature)
                    else:
                        categorical_features.append(feature)

            self.logger.log(self.file_object, "separate_cat_num Completed!!!")
            return categorical_features, numerical_features

        except Exception as e:
            self.logger.log(self.file_object, "Cannot Separate Categorical and Numerical Variables: %s"%Exception(e))
            raise Exception(e)

    def find_statistical_least_significant_variable(self,df,categorical_features,numerical_features,target):
        """
        Method Name: find_statistical_least_significant_variable
        Description: This function tries to find out the least significant variables from the dataset through some
                     statistical hypothesis and experiments done for each variable against the target variable

                     (i) For categorical variable: we will do chi-square test between the independant variables
                      and the target variable.This p-Value calculated will help us to understand the significance
                      of the variable over target variable

                     (ii) For numerical variable: we will perform Independant Sample T Test and check the p-Value
                     to understand which variable has better explainatory power and which has the least.

                     All the least significant variiables will be dumped into an array and passed as an output.

                     Here we will consider the acceptable p-Value as 0.05 with 95% confidence level.


        Parameter: df(the dataframe), categorical_features, numerical_features, target
        Output: Least Significant variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "find_statistical_least_significant_variable Started!!!")
        least_sigVar_list = []

        try:
            chisq_df = pd.DataFrame() #Empty dataframe
            for feature in categorical_features:
                cross_tab = pd.crosstab(df[feature], target, margins=False)
                tmp = pd.DataFrame([feature, stats.chi2_contingency(observed=cross_tab)[0],
                                    stats.chi2_contingency(observed=cross_tab)[1]]).T
                tmp.columns = ['Feature', 'Chi-Square', 'p-Value']
                chisq_df = pd.concat([chisq_df, tmp], axis=0, ignore_index=True)

            # convert the dataframe to numpy array
            ndArrayList = chisq_df.to_numpy()

            #loop through the list and find out features which has p-Value more than 0.05
            for list_ in ndArrayList:
                #list_[0] -> Feature, list_[1] -> Chi-Square Value , list_[2] -> p-Value
                if list_[2] > 0.05:
                    least_sigVar_list.append(list_[0]) #feature name is added to the list


            tstats_df = pd.DataFrame()  # Empty dataframe
            df1 = pd.concat([df,target],axis=1)
            for feature in numerical_features:
                tstats = stats.ttest_ind(df1[df1['target'] == 1][feature],
                                     df1[df1['target'] == 0][feature])
                temp = pd.DataFrame([feature, tstats[0], tstats[1]]).T
                temp.columns = ['Feature', 'T-Statistics', 'p-Value']
                tstats_df = pd.concat([tstats_df, temp], axis=0, ignore_index=True)

            # convert the dataframe to numpy array
            ndArrayList1 = tstats_df.to_numpy()

            # loop through the list and find out features which has p-Value more than 0.05
            for list_ in ndArrayList1:
                if list_[2] > 0.05:
                    # list_[0] -> Feature, list_[1] -> T-Statistics Value , list_[2] -> p-Value
                    least_sigVar_list.append(list_[0])

            self.logger.log(self.file_object, "find_statistical_least_significant_variable Completed!!!")
            return least_sigVar_list

        except Exception as e:
            self.logger.log(self.file_object, "Error finding statistically least significant variables due to : %s" % Exception(e))
            raise Exception(e)

    def checkforMultiCollinearity(self, df, threshold):
        """
        Method Name: checkforMultiCollinearity
        Description: This function find out the variables which have high multi-collinearity among themselves.

        Parameter: df(the dataframe), threshold(cut off value till what multi-collinearity is accepted)
        Output: List of high multi-collinear variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "checkforMultiCollinearity Started!!!")
        try:
            corr_set = set()
            corr_matrix = df.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > threshold:
                        colName = corr_matrix.columns[i]
                        corr_set.add(colName)

            self.logger.log(self.file_object, "checkforMultiCollinearity Completed!!!")
            return list(corr_set)

        except Exception as e:
            self.logger.log(self.file_object, "Finding Multi-Collinearity Failed: %s"%Exception(e))
            raise Exception(e)

    def scaleData(self, data):
        """
        Method Name: scaleData
        Description: This function scales the dataset in same unit using StandardScaler

        Parameter: data(the dataframe)
        Output: List of scaled variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "scaleData method Started!!!")
        self.data = data
        try:
            scaler = StandardScaler()
            arr = scaler.fit_transform(self.data)
            self.logger.log(self.file_object, "scaleData method Completed!!!")
            return arr
        except Exception as e:
            self.logger.log(self.file_object, "StandardScaler Conversion Failed: %s" % Exception(e))
            raise Exception(e)

    def scaleDataTransform(self,data):
        """
        Method Name: scaleDataTransform
        Description: This function scales the dataset in same unit using StandardScaler

        Parameter: data(the dataframe)
        Output: dataframe
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "scaleDataTransform method Started!!!")
        self.data = data
        try:
            scaler = StandardScaler()
            arr = scaler.fit_transform(self.data)
            self.logger.log(self.file_object, "scaleDataTransform scaler Completed!!!")
            df = pd.DataFrame(data=arr, columns=self.data.columns)

            self.logger.log(self.file_object, "scaleDataTransform method Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "StandardScaler Conversion Failed: %s" % Exception(e))
            raise Exception(e)












