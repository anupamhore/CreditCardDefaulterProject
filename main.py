from flask import Flask, request, render_template, redirect,url_for
from flask import Response
import os
from flask_cors import  CORS, cross_origin
from trainingValidation import train_validation
from predictionValidation import predictValidation
from trainingModel import TrainModel
from predictionModel import PredictModel
from datetime import datetime
import json
import pandas as pd

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.form is not None:
            try:
                if 'file' not in request.files:
                    print('File not found')
                    return render_template('index.html', results="Problem in locating the file!!!")
                else:
                    file = request.files['file']
                    if file:
                        startTime = datetime.now().replace(microsecond=0)

                        pred_val_obj = predictValidation(file)
                        isValidationSuccess, endTime1 = pred_val_obj.startValidation()


                        if isValidationSuccess:

                            pred_model = PredictModel()
                            endTime = pred_model.predictTestData()

                            totalDiff = endTime - startTime

                            df = pd.read_csv('Prediction_Output_File/Predictions.csv')
                            df.drop('Unnamed: 0',axis=1,inplace=True)
                            html = df.to_html()


                            return render_template('index.html',results="See the predictions as below:" ,results1="Total Execution Time:%s" % totalDiff, tables=[html], titles=df.columns.values)
                        else:
                            totalDiff1 = endTime1 - startTime
                            return render_template('index.html', results="The test file should follow the same format as the one mentioned in the Original Source of Data below",
                                                   results1="Total Execution Time:%s" % totalDiff1)

            except Exception as e:
                print("Error: %s"%Exception(e))
                return render_template('index.html', results=Exception(e))

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['filepath'] is not None:
            path = request.json['filepath']

            train_val_obj = train_validation(path)
            train_val_obj.startValidation()

            trainingModelObj = TrainModel()
            trainingModelObj.modelTraining()


    except ValueError:
        return Response("Error Occurred! %s" % ValueError)

    except KeyError:
        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:
        return Response("Error Occurred! %s" % e)

    return Response("Training Successfull!!")


if __name__ == "__main__":
    app.run(debug=True)


