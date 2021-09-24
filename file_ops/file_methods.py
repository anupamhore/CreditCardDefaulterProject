import pickle
import os
import shutil

class File_Operation:

    def __init__(self):
        self.model_directory = 'models/'

    def save_model(self, model, filename):

        try:
            path = os.path.join(self.model_directory, filename)

            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)

            with open(path + "/" + filename + '.sav', 'wb') as f:
                pickle.dump(model, f)

            return 'success'

        except Exception as e:
            raise Exception(e)



    def load_model(self, filename):

        try:
            with open(self.model_directory + filename + "/" + filename + '.sav', 'rb') as f:
                return pickle.load(f)

        except Exception as e:
            raise Exception(e)

    def find_correct_model_file(self,cluster_number):

        try:
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_model_files  = []
            self.list_of_files = os.listdir(self.folder_name)

            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name=self.file
                except:
                    continue
            self.model_name = self.model_name.split('.')[0]

            return self.model_name

        except Exception as e:
            raise Exception(e)
