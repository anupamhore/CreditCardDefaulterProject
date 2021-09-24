import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from file_ops import file_methods

#To show the graphs which are not called from main thread
plt.switch_backend('Agg')

class KMeansClustering:
    """
        This class shall  be used to create clusters for the dataset

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
    """
    def __init__(self, logger):
        self.logger = logger
        self.file_object = open('Training_Logs/KmeansClustering.txt', 'a+')


    def elbow_plot(self, data):
        """
        Method Name: elbow_plot
        Description: This function plots the points on the graph.The shape of the graph is like
                     an elbow. This point at which the elbow is formed is normally taken as the
                     number of clusters for that dataset.

                     We also used an automatic method of KneeLocator to automatically find us
                     the number of clusters for the given dataset.

        Parameter: data
        Output: Total Number of clusters
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"elbow_plot Started!!!")
        wcss = []
        try:
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1, 11), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('data_preprocessing/K-Means_Elbow.PNG')

            #finding the value of the optimum cluster programatically
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger.log(self.file_object, "elbow_plot Completed!!!")
            return self.kn.knee

        except Exception as e:
            self.logger.log(self.file_object, "Problem in figuring out the clusters %s"%Exception(e))
            raise Exception(e)

    def create_clusters(self, data, number_of_clusters):
        """
        Method Name: create_clusters
        Description: This function creates the different clusters for the dataset and update the dataframe

        Parameter: data, number_of_clusters
        Output: Returns the updated dataframe with the cluster number as a new column
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "create_clusters Started!!!")
        self.data = data

        try:
            self.kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            self.pred_y_kmeans = self.kmeans.fit_predict(self.data)

            self.file_op = file_methods.File_Operation()
            self.save_model = self.file_op.save_model(self.kmeans, 'KMeans')

            self.df = pd.DataFrame(self.data)
            self.df['Cluster'] = self.pred_y_kmeans

            self.logger.log(self.file_object, "create_clusters Completed!!!")
            self.file_object.close()
            return self.df

        except Exception as e:
            self.logger.log(self.file_object, "Problem in creating the clusters %s" % Exception(e))
            self.file_object.close()
            raise Exception(e)

