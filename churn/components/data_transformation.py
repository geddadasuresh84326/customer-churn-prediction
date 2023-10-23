import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


from churn.constant.training_pipeline import TARGET_COLUMN
from churn.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from churn.entity.config_entity import DataTransformationConfig
from churn.exception import ChurnException
from churn.logger import logging
# from churn.ml.model.estimator import TargetValueMapping
from churn.utils.main_utils import save_numpy_array_data, save_object




class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise ChurnException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ChurnException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    ("Imputer", simple_imputer), #replace missing values with zero
                    ("RobustScaler", robust_scaler) #keep every feature in same range and handle outlier
                    ]
            )
            
            return preprocessor

        except Exception as e:
            raise ChurnException(e, sys) from e
    @classmethod
    def label_encoding(cls,dataframe:pd.DataFrame)->pd.DataFrame:
        cat_columns = [feature for feature in dataframe.columns if dataframe[feature].dtype == "O"]
        logging.info(f"cateforical_columns : {cat_columns}")
        for feature in cat_columns:
            dummies = pd.get_dummies(dataframe[feature])
            print(dummies.head(2))
            dataframe = pd.concat([dummies,dataframe],axis = 1)
            dataframe.drop(feature,axis= 1,inplace= True)
        return dataframe
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()

            # label encoding 
            train_labeled_df = DataTransformation.label_encoding(dataframe=train_df)
            test_labeled_df = DataTransformation.label_encoding(dataframe=test_df)
            
            #training dataframe
            input_feature_train_df = train_labeled_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_labeled_df[TARGET_COLUMN]

            #testing dataframe
            input_feature_test_df = test_labeled_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_labeled_df[TARGET_COLUMN]

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)


            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ChurnException(e, sys) from e

