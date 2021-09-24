from flask import Flask, request, render_template, redirect,url_for
from flask import Response
import os
from flask_cors import  CORS, cross_origin
from trainingValidation import train_validation
from predictionValidation import predictValidation
from trainingModel import TrainModel
from predictionModel import PredictModel
from datetime import datetime


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
                        pred_val_obj.startValidation()

                        pred_model = PredictModel()
                        path,endTime = pred_model.predictTestData()

                        totalDiff = endTime - startTime


                return render_template('index.html',
                                       results="Prediction File created at: %s"%path,
                                       results1="Total Execution Time:%s"%totalDiff)

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
    app.run(debug=False)


