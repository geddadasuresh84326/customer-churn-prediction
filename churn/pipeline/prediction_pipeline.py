from churn.exception import ChurnException
from churn.logger import logging
import os,sys
from churn.utils.main_utils import read_yaml_file,load_object
from churn.constant.prediction_pipeline import SCHEMA_FILE_PATH,SCHEMA_DROP_COLS
import pandas as pd
from churn.components.data_transformation import DataTransformation
from churn.entity.config_entity import PredictionPipelineConfig

class Prediction:
    def __init__(self,path):
        try:
            logging.info("Prediction pipeline started")
            self.prediction_file_path = path
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = PredictionPipelineConfig()
            logging.info(f"Prediction file path : {self.prediction_file_path}")
        except Exception as e:
            raise ChurnException(e,sys)
        
    def prediction_file_validation(self)->pd.DataFrame:
        try:
            logging.info("Inside prediction_file_validation method ")
            df = pd.read_csv(self.prediction_file_path)
            schema_columns = self._schema_config["columns"]
            logging.info(f"prediction file columns : {df.columns}")
            if len(df.columns) != len(schema_columns):
                raise Exception(f"Prediction file columns as not matched .it has {len(df.columns)} but schema as {len(schema_columns)}")
            return df
        except Exception as e:
            raise ChurnException(e,sys)

    def prediction_file_transformation(self,dataframe:pd.DataFrame):
        try:
            logging.info("Inside prediction_file_transformation")
            logging.info("dropping unnecessary columns started")
            customer_df = list(dataframe['CustomerID'])
            dataframe = dataframe.drop(self._schema_config['drop_columns'],axis= 1)
            logging.info(f" unnecessary columns dropped : {self._schema_config['drop_columns']}")
            logging.info(f"label encoding started")
            logging.info(f"column names are : {dataframe.columns}")
            df = DataTransformation.label_encoding(dataframe=dataframe)
            logging.info(f"label encoding completed and df columns are : {df.columns}")

            logging.info(f"predictions started")
            # getting the model.pkl from saved_models folder
            maxtimestamp = max(list(map(int,os.listdir("saved_models"))))
            model_path = os.path.join("saved_models",f"{maxtimestamp}","model.pkl")
            model_and_preprocessor = load_object(model_path)
            predictions = list(model_and_preprocessor.predict(df))
            logging.info(f"prediction completed")

            result = pd.DataFrame(list(zip(customer_df,predictions)),columns=["CustomerID","Churn"])
            dir_path = os.path.dirname(self.prediction_pipeline_config.prediction_result_file_path)
            logging.info(f"saving result into : {dir_path}")

            os.makedirs(dir_path,exist_ok=True)
            result.to_csv("prediction_result\prediction.csv",index=False)

            logging.info(f"Predictions saved successfully...!")

            return dir_path,result            
        except Exception as e:
            raise ChurnException(e,sys)

    def initiate_prediction(self):
        df = self.prediction_file_validation()
        return self.prediction_file_transformation(dataframe=df)
