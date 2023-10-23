from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
from flask.json import jsonify
import os
from flask_cors import CORS, cross_origin
# import flask_monitoringdashboard as dashboard
from churn.pipeline.prediction_pipeline import Prediction
import json
import io
from churn.pipeline.prediction_pipeline import Prediction
import pandas as pd

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
# dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = Prediction(path=path) #object initialization

            pred_val.prediction_file_validation() #calling the prediction_validation function

            # pred = prediction(path) #object initialization

            # predicting for dataset present in database
            # path,json_predictions = pred.predictionFromModel()
            # return Response("Prediction File created at !!!"  +str(path) +'and few of the predictions are '+str(json.loads(json_predictions) ))
        elif request.form is not None:
            path = request.form['filepath']

            pred_val = Prediction(path) #object initialization

            path,json_predictions = pred_val.initiate_prediction() #calling the initiate_prediction function

            return Response('Few of the predictions are : '+str(json.loads(json_predictions.head().to_json(orient="records")) ))
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)


# @app.route("/train", methods=['POST'])
# @cross_origin()
# def trainRouteClient():

#     try:
#         if request.json['folderPath'] is not None:
#             path = request.json['folderPath']

#             train_valObj = train_validation(path) #object initialization

#             train_valObj.train_validation()#calling the training_validation function


#             trainModelObj = trainModel() #object initialization
#             trainModelObj.trainingModel() #training the model for the files in the table


#     except ValueError:

#         return Response("Error Occurred! %s" % ValueError)

#     except KeyError:

#         return Response("Error Occurred! %s" % KeyError)

#     except Exception as e:

#         return Response("Error Occurred! %s" % e)
#     return Response("Training successfull!!")


@app.route('/download-csv', methods=['GET'])
def download_csv():
    df = pd.read_csv("prediction_result\prediction.csv")
    # Create a buffer to store the CSV content
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Serve the CSV file as a downloadable response
    response = Response(csv_buffer, mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=data.csv'
    return response

@app.route('/upload', methods=['POST'])
def upload_csv():
    csv_file = request.files['csv-file']
    if csv_file:
        pred_val = Prediction(csv_file) #object initialization

        path,json_predictions = pred_val.initiate_prediction() #calling the initiate_prediction function

        return Response('Custom file predicitons Few of the predictions are : '+str(json.loads(json_predictions.head().to_json(orient="records")) ))

    return Response('error No file provided')

port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
