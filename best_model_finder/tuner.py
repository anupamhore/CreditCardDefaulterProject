from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score
from collections import Counter
from imblearn.combine import SMOTETomek

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('always')

class Model_Finder:
    """
        This class will help us to find out the best model for the given cluster

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
    """
    def __init__(self,logger):
        self.logger = logger
        self.file_object = open('Training_Logs/ModelFinder.txt','a+')

    def get_best_params_for_logisticRegression(self,X_train, y_train):
        """
          Method Name: get_best_params_for_logisticRegression
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "Logistic Regression Started!!!")
        try:
            log_reg = LogisticRegression()

            param_grid = {
                "penalty":["l1", "l2", "elasticnet", "none"],
                "C":np.logspace(-4,4,20),
                "solver":["newton-cg", "lbfgs", "sag", "saga"],
                "max_iter":[100,1000,2500,5000]
            }
            clf = GridSearchCV(log_reg,param_grid=param_grid,cv=5,verbose=False,n_jobs=-1)
            best_clf = clf.fit(X_train, y_train)

            self.logger.log(self.file_object, "Logistic Regression Completed!!!")
            return best_clf

        except Exception as e:
            self.logger.log(self.file_object, "Logistic Regression Model Error: %s"%Exception(e))
            raise Exception(e)

    def get_best_params_for_DecisionTree(self,X_train, y_train):
        """
          Method Name: get_best_params_for_DecisionTree
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "Decision Tree Started!!!")
        try:

            dt = DecisionTreeClassifier()

            #First we will perform randomized search CV. From here we will get the range to use for GridSearchCV
            random_grid = {
                "criterion": ["gini", "entropy"],
                "splitter": ["best","random"],
                "max_depth": [int(x) for x in np.linspace(10, 1000,10)],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [3,4,5,7,9,13],
                "min_samples_leaf":  [2,4,6,8],
                "ccp_alpha":np.arange(0, 1, 0.001).tolist()
            }

            clf_random = RandomizedSearchCV(dt, param_distributions=random_grid,
                                            n_iter=100, cv= 5, verbose=False, random_state=101, n_jobs=-1)
            best_clf_random = clf_random.fit(X_train, y_train)

            param_grid = {
                "criterion": [best_clf_random.best_params_["criterion"]],
                "splitter": [best_clf_random.best_params_["splitter"]],
                "max_depth": [best_clf_random.best_params_["max_depth"]],
                "max_features": [best_clf_random.best_params_["max_features"]],
                "min_samples_split": [best_clf_random.best_params_["min_samples_split"] - 2,
                                      best_clf_random.best_params_["min_samples_split"] - 1,
                                      best_clf_random.best_params_["min_samples_split"],
                                      best_clf_random.best_params_["min_samples_split"] + 1,
                                      best_clf_random.best_params_["min_samples_split"] + 2],
                "min_samples_leaf": [best_clf_random.best_params_["min_samples_leaf"],
                                     best_clf_random.best_params_["min_samples_leaf"] + 2,
                                     best_clf_random.best_params_["min_samples_leaf"] + 4],
                "ccp_alpha": [best_clf_random.best_params_["ccp_alpha"]]
            }

            clf_grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv =5, verbose=False,n_jobs=-1)
            best_clf = clf_grid.fit(X_train, y_train)

            self.logger.log(self.file_object, "Decision Tree Completed!!!")

            return best_clf

        except Exception as e:
            self.logger.log(self.file_object, "Decision Tree Error: %s"%Exception(e))
            raise Exception(e)

    def get_best_params_for_BaggingClassifier(self,X_train, y_train, clf_df, X_test, y_test):
        """
           Method Name: get_best_params_for_BaggingClassifier
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "Bagging Classifier Started!!!")
        try:

            param_grid = {
                'n_estimators':  [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                'base_estimator__max_leaf_nodes': [clf_df.best_params_["min_samples_leaf"],
                                                   clf_df.best_params_["min_samples_leaf"] + 2,
                                                   clf_df.best_params_["min_samples_leaf"] + 4],
                'base_estimator__max_depth': [clf_df.best_params_["max_depth"]]
            }

            bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), oob_score=True,random_state=1)

            clf = GridSearchCV(estimator=bagging, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
            best_clf = clf.fit(X_train, y_train)

            self.logger.log(self.file_object,
                            "Bagging Classifier Best Grid Params:%s" %best_clf.best_params_)


            self.logger.log(self.file_object,
                            "Bagging Classifier GridSearch Score(Train Data):%s" %clf.score(X_train, y_train))
            self.logger.log(self.file_object,
                            "Bagging Classifier GridSearch Score(Test Data):%s" %clf.score(X_test, y_test))
            #
            # final_dt = DecisionTreeClassifier(max_leaf_nodes=10, max_depth=5)
            # final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=40, random_state=1, oob_score=True)


            self.logger.log(self.file_object, "Bagging Classifier Completed!!!")

            return best_clf


        except Exception as e:
            self.logger.log(self.file_object, "Bagging Classifier Error: %s"%Exception(e))
            raise Exception(e)

    def get_best_params_for_RandomForestClassifier(self,X_train, y_train):
        """
          Method Name: get_best_params_for_RandomForestClassifier
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "RandomForest Classifier Started!!!")
        try:
            rf = RandomForestClassifier()
            random_grid = {
                'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                "criterion": ["gini", "entropy"],
                "max_depth": [int(x) for x in np.linspace(10, 1000,10)],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [3,4,5,7,9,13],
                "min_samples_leaf":  [2,4,6,8],
                "ccp_alpha":np.arange(0, 1, 0.001).tolist()
            }
            clf_random = RandomizedSearchCV(rf,param_distributions=random_grid,cv=5,
                                            verbose=False,n_jobs=-1,n_iter=100,random_state=101)

            best_clf_random = clf_random.fit(X_train, y_train)

            param_grid = {
                "criterion": [best_clf_random.best_params_["criterion"]],
                "max_depth": [best_clf_random.best_params_["max_depth"]],
                "max_features": [best_clf_random.best_params_["max_features"]],
                "min_samples_split": [best_clf_random.best_params_["min_samples_split"] - 2,
                                      best_clf_random.best_params_["min_samples_split"] - 1,
                                      best_clf_random.best_params_["min_samples_split"],
                                      best_clf_random.best_params_["min_samples_split"] + 1,
                                      best_clf_random.best_params_["min_samples_split"] + 2],
                "min_samples_leaf": [best_clf_random.best_params_["min_samples_leaf"],
                                     best_clf_random.best_params_["min_samples_leaf"] + 2,
                                     best_clf_random.best_params_["min_samples_leaf"] + 4],
                "ccp_alpha": [best_clf_random.best_params_["ccp_alpha"]],
                "n_estimators": [best_clf_random.best_params_['n_estimators'] - 200,
                                 best_clf_random.best_params_['n_estimators'] - 100,
                                 best_clf_random.best_params_['n_estimators'],
                                 best_clf_random.best_params_['n_estimators'] + 100,
                                 best_clf_random.best_params_['n_estimators'] + 200]
            }

            clf_grid = GridSearchCV(RandomForestClassifier(),param_grid=param_grid, cv = 5, verbose=False,n_jobs=-1)
            best_clf = clf_grid.fit(X_train, y_train)
            self.logger.log(self.file_object, "RandomForest Classifier Completed!!!")

            return best_clf


        except Exception as e:
            self.logger.log(self.file_object, "RandomForest Classifier Error: %s"%Exception(e))
            raise Exception(e)

    def get_best_params_for_AdaBoostingClassifier(self,X_train, y_train):
        """
          Method Name: get_best_params_for_AdaBoostingClassifier
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "AdaBoosting Classifier Started!!!")
        try:
            adaboost_clf = AdaBoostClassifier()

            param_grid = {
                'n_estimators': [10, 50, 100, 500],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
            }
            clf = GridSearchCV(adaboost_clf,param_grid=param_grid, cv = 5, verbose=False, n_jobs=-1)
            best_clf = clf.fit(X_train, y_train)

            self.logger.log(self.file_object, "AdaBoosting Classifier Completed!!!")

            return  best_clf


        except Exception as e:
            self.logger.log(self.file_object, "AdaBoosting Classifier Error: %s" %Exception(e))
            raise Exception(e)

    def get_best_params_for_GradientBoostingClassifier(self,X_train, y_train):
        """
          Method Name: get_best_params_for_GradientBoostingClassifier
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "Gradient Boosting Classifier Started!!!")
        try:
            gb = GradientBoostingClassifier()

            random_grid = {
                'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'max_features': ['auto', 'sqrt','log2'],
                #'criterion': ['friedman_mse', 'mse'],
                'max_depth': [int(x) for x in np.linspace(10, 1000,10)],
                'min_samples_split': [3,4,5,7,9,13],
                'min_samples_leaf':  [2,4,6,8],
                'subsample': [0.7,0.8,0.9,1.0],
                'warm_start': [True,False],
                #"ccp_alpha": np.arange(0, 1, 0.001).tolist()
            }

            clf_random = RandomizedSearchCV(gb,param_distributions=random_grid, cv=5,
                                            verbose=False,n_jobs=-1,n_iter=100,random_state=101)


            best_clf_random = clf_random.fit(X_train, y_train)

            param_grid = {
                #"criterion": [best_clf_random.best_params_["criterion"]],
                "max_depth": [best_clf_random.best_params_["max_depth"]],
                "max_features": [best_clf_random.best_params_["max_features"]],
                "learning_rate":[best_clf_random.best_params_["learning_rate"]],
                "subsample":[best_clf_random.best_params_["subsample"]],
                "min_samples_split": [best_clf_random.best_params_["min_samples_split"] - 2,
                                      best_clf_random.best_params_["min_samples_split"] - 1,
                                      best_clf_random.best_params_["min_samples_split"],
                                      best_clf_random.best_params_["min_samples_split"] + 1,
                                      best_clf_random.best_params_["min_samples_split"] + 2],
                "min_samples_leaf": [best_clf_random.best_params_["min_samples_leaf"],
                                     best_clf_random.best_params_["min_samples_leaf"] + 2,
                                     best_clf_random.best_params_["min_samples_leaf"] + 4],
                # "ccp_alpha": [best_clf_random.best_params_["ccp_alpha"]],
                "n_estimators": [best_clf_random.best_params_['n_estimators'] - 200,
                                 best_clf_random.best_params_['n_estimators'] - 100,
                                 best_clf_random.best_params_['n_estimators'],
                                 best_clf_random.best_params_['n_estimators'] + 100,
                                 best_clf_random.best_params_['n_estimators'] + 200],
            }

            clf_grid = GridSearchCV(GradientBoostingClassifier(),param_grid=param_grid, cv = 5, verbose=False,n_jobs=-1)
            best_clf = clf_grid.fit(X_train, y_train)

            self.logger.log(self.file_object, "Gradient Boosting Classifier Completed!!!")

            return best_clf

        except Exception as e:
            self.logger.log(self.file_object, "Gradient Boosting Classifier Error: %s"%Exception(e))
            raise Exception(e)

    def get_best_params_for_XtremeGradientBoostingClassifier(self,X_train, y_train):
        """
          Method Name: get_best_params_for_XtremeGradientBoostingClassifier
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "Extreme Gradient Boosting Classifier Started!!!")
        try:
            xgb = XGBClassifier(objective='binary:logistic',use_label_encoder=False,eval_metric='auc')

            param_grid = {
                "n_estimators": [50,100, 130],
                "max_depth": range(3, 11, 1),
                "random_state": [0, 50, 100]
            }

            clf = GridSearchCV(xgb,param_grid=param_grid, cv=3, verbose=False,n_jobs=-1)
            best_clf = clf.fit(X_train, y_train)

            xbg_train = XGBClassifier(random_state=clf.best_params_['random_state'],max_depth=clf.best_params_['max_depth'],
                                n_estimators=clf.best_params_['n_estimators'],n_jobs=-1)

            xbg_train.fit(X_train,y_train)


            self.logger.log(self.file_object, "Extreme Gradient Boosting Classifier Completed!!!")

            return xbg_train

        except Exception as e:
            self.logger.log(self.file_object, "Extreme Gradient Boosting Error: %s"%Exception(e))
            raise Exception(e)

    def get_best_params_for_NaiveBayesClassifier(self,X_train, y_train):
        """
          Method Name: get_best_params_for_NaiveBayesClassifier
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "Naive Bayes Classifier Started!!!")
        try:
            gnb = GaussianNB()
            param_grid = {
                "var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]
            }

            clf = GridSearchCV(gnb,param_grid=param_grid, cv=5,verbose=False,n_jobs=-1)
            best_clf = clf.fit(X_train, y_train)

            self.logger.log(self.file_object, "Naive Bayes Classifier Completed!!!")
            return best_clf

        except Exception as e:
            self.logger.log(self.file_object, "Naive Bayes Classifier Error: %s"%Exception(e))
            raise Exception(e)

    def get_best_params_for_SVC(self,X_train, y_train):
        """
          Method Name: get_best_params_for_SVC
          Description:This class trains the data based on hyper parameter tuning
          Input: X_train, y_train
          Output: model name, model itself
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "SVM Classifier Started!!!")
        try:
            os = SMOTETomek(0.5)
            X_train_svc, y_train_svc = os.fit_resample(X_train,y_train)
            svc = SVC()
            param_grid = {
                'C': [0.1, 1, 100, 1000],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'degree': [1, 2, 3, 4, 5, 6]
            }

            clf = GridSearchCV(svc,param_grid=param_grid, cv=5,n_jobs=-1, verbose=False,scoring='roc_auc')
            best_clf = clf.fit(X_train_svc, y_train_svc)

            self.logger.log(self.file_object, "SVM Classifier Completed!!!")

            return best_clf

        except Exception as e:
            self.logger.log(self.file_object, "SVM Classifier Error: %s"%Exception(e))
            raise Exception(e)




    def getModelScore(self, model, X, Y, modelName):

        """
          Method Name: getModelScore
          Description:This class calculate the model score based on y_pred, y_test
          Input: X,Y
          Output: score of the model
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, "getModelScore Started!!!")
        try:
            y_pred = model.predict(X)
            score = 0
            if len(Y.unique()) == 1:
                score = accuracy_score(Y, y_pred)
            else:
                score = roc_auc_score(Y,y_pred)

            self.logger.log(self.file_object,'{} roc_auc_score: {}'.format(modelName,score))
            # self.logger.log(self.file_object, "Classification Report for %s"%modelName)
            # self.logger.log(self.file_object, "%s" % classification_report(Y, y_pred))
            self.logger.log(self.file_object, "getModelScore Completed!!!")
            return score

        except Exception as e:
            self.logger.log(self.file_object, "getModelScore Error: %s" % Exception(e))
            raise Exception(e)


    def get_best_model(self,X_train, y_train, X_test, y_test):
        """
        Method Name: get_best_model
        Description: This function will give us the best model for the given cluster. We will create
                     various model, perform various hyper parameter tuning and then select the model
                     which gives the best accuracy. We will also test the model with the test data.
                     If required we will perform hyper parameter tuning on the selected model again

        Parameter: X_train, y_train, X_test, y_test
        Output: Gives the best model name and the model itself
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"get_best_model Started!!!")
        try:

            # Logistic Regression
            # self.best_clf_log_reg = self.get_best_params_for_logisticRegression(X_train, y_train)
            # self.logger.log(self.file_object,
            #                 "Logistic Regression GridSearch Score(Train Data):%s" %self.best_clf_log_reg.score(X_train, y_train))
            # self.logger.log(self.file_object,
            #                 "Logistic Regression GridSearch Score(Test Data):%s" %self.best_clf_log_reg.score(X_test, y_test))
            #
            # self.log_reg = self.best_clf_log_reg.best_estimator_
            # self.log_reg_score = self.getModelScore(self.log_reg, X_test, y_test,'LogisticRegression')


            #Decision Tree
            # self.best_clf_dt = self.get_best_params_for_DecisionTree(X_train,y_train)
            # self.logger.log(self.file_object,
            #                 "Decision Tree GridSearch Score(Train Data):%s" %self.best_clf_dt.score(X_train, y_train))
            # self.logger.log(self.file_object,
            #                 "Decision Tree GridSearch Score(Test Data):%s" %self.best_clf_dt.score(X_test, y_test))
            #
            #
            # self.dt = self.best_clf_dt.best_estimator_
            # self.dt_score = self.getModelScore(self.dt, X_test, y_test,'Decision Tree')


            #Bagging
            # self.best_clf_bagging = self.get_best_params_for_BaggingClassifier(X_train,y_train,self.best_clf_dt, X_test, y_test)
            #
            #
            # self.bagging_classifier = self.best_clf_bagging.best_estimator_
            # self.bagging_classifier_score = self.getModelScore(self.bagging_classifier, X_test, y_test,'Bagging Classifier')


            #RandomForest
            self.best_clf_rf = self.get_best_params_for_RandomForestClassifier(X_train,y_train)
            self.logger.log(self.file_object,
                            "RandomForest Classifier GridSearch Score(Train Data):%s" %self.best_clf_rf.score(X_train, y_train))
            self.logger.log(self.file_object,
                            "RandomForest Classifier GridSearch Score(Test Data):%s" %self.best_clf_rf.score(X_test, y_test))

            self.rf = self.best_clf_rf.best_estimator_
            self.rf_classifier_score = self.getModelScore(self.rf, X_test, y_test, 'RandomForest Classifier')


            # #Ada Boosting
            # self.best_clf_adaboosting = self.get_best_params_for_AdaBoostingClassifier(X_train,y_train)
            # self.logger.log(self.file_object,
            #                 "AdaBoosting Classifier GridSearch Score(Train Data):%s" %self.best_clf_adaboosting.score(X_train, y_train))
            # self.logger.log(self.file_object,
            #                 "AdaBoosting Classifier GridSearch Score(Test Data):%s" %self.best_clf_adaboosting.score(X_test, y_test))
            #
            # self.adaboosting = self.best_clf_adaboosting.best_estimator_
            # self.adaboosting_classifier_score = self.getModelScore(self.adaboosting, X_test, y_test, 'AdaBoosting Classifier')


            #Gradient Boosting
            # self.best_clf_gb = self.get_best_params_for_GradientBoostingClassifier(X_train,y_train)
            # self.logger.log(self.file_object,
            #                 "Gradient Boosting Classifier GridSearch Score(Train Data):%s" %self.best_clf_gb.score(X_train, y_train))
            # self.logger.log(self.file_object,
            #                 "Gradient Boosting Classifier GridSearch Score(Test Data):%s" %self.best_clf_gb.score(X_test, y_test))
            #
            # self.gb = self.best_clf_gb.best_estimator_
            # self.gb_classifier_score = self.getModelScore(self.gb, X_test, y_test, 'Gradient Boosting Classifier')


            #Xtreme Gradient Boosting
            self.best_clf_xgb = self.get_best_params_for_XtremeGradientBoostingClassifier(X_train,y_train)
            self.logger.log(self.file_object,
                            "Extreme Gradient Boosting Classifier GridSearch Score(Train Data):%s" %self.best_clf_xgb.score(X_train, y_train))
            self.logger.log(self.file_object,
                            "Extreme Gradient Boosting Classifier GridSearch Score(Test Data):%s" %self.best_clf_xgb.score(X_test, y_test))

            self.xgb = self.best_clf_xgb
            self.xgb_classifier_score = self.getModelScore(self.xgb, X_test, y_test, 'Extreme Gradient Boosting Classifier')

            #Naive Bayes
            self.best_clf_nb = self.get_best_params_for_NaiveBayesClassifier(X_train,y_train)
            self.logger.log(self.file_object,
                            "Naive Bayes Classifier GridSearch Score(Train Data):%s" %self.best_clf_nb.score(X_train, y_train))
            self.logger.log(self.file_object,
                            "Naive Bayes Classifier GridSearch Score(Test Data):%s" %self.best_clf_nb.score(X_test, y_test))

            self.nb = self.best_clf_nb.best_estimator_
            self.nb_classifier_score = self.getModelScore(self.nb, X_test, y_test, 'Naive Bayes Classifier')


            #Support Vector Machine(SVC)
            # self.best_clf_svc = self.get_best_params_for_SVC(X_train,y_train)
            # self.logger.log(self.file_object,
            #                 "SVC GridSearch Score(Train Data):%s" %self.best_clf_svc.score(X_train, y_train))
            # self.logger.log(self.file_object,
            #                 "SVC GridSearch Score(Test Data):%s" %self.best_clf_svc.score(X_test, y_test))
            #
            # self.svc = self.best_clf_svc.best_estimator_
            # self.svc_classifier_score = self.getModelScore(self.svc, X_test, y_test, 'SVC')

            self.scoreList = [
                # {"modelName": "Logistic Regression", "modelscore": self.log_reg_score, "model": self.log_reg},
                # {"modelName": "Decision Tree", "modelscore": self.dt_score,"model": self.dt},
                # {"modelName": "Bagging", "modelscore": self.bagging_classifier_score, "model": self.bagging_classifier},
                {"modelName": "Random Forest", "modelscore": self.rf_classifier_score, "model": self.rf},
                # {"modelName": "Ada Boosting", "modelscore": self.adaboosting_classifier_score, "model": self.adaboosting},
                # {"modelName": "Gradient Boosting", "modelscore": self.gb_classifier_score, "model": self.gb},
                {"modelName": "Extreme Gradient Boosting", "modelscore": self.xgb_classifier_score, "model": self.xgb},
                {"modelName": "Naive Bayes", "modelscore": self.nb_classifier_score, "model": self.nb},
                # {"modelName": "SVC", "modelscore": self.svc_classifier_score, "model": self.svc},
            ]
            self.scoreList.sort(key=lambda x: x['modelscore'], reverse=True)
            modelObject = self.scoreList[0]

            self.logger.log(self.file_object, "get_best_model Completed!!!")

            return modelObject['modelName'], modelObject['model']

        except Exception as e:
            self.logger.log(self.file_object, "Problem getting best model %s"%Exception(e))
            raise Exception(e)
