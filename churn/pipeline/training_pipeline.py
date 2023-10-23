from churn.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig,ModelPusherConfig
from churn.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact,ModelPusherArtifact
from churn.components.data_ingestion import DataIngestion
from churn.components.data_validation import DataValidation
from churn.components.data_transformation import DataTransformation
from churn.components.model_trainer import ModelTrainer
from churn.components.model_evaluation import ModelEvaluation
from churn.components.model_pusher import ModelPusher
from churn.exception import ChurnException
from churn.logger import logging
import os,sys

class TrainPipeline:
    is_pipeline_running = False
    def __init__(self) :
        training_pipeline_config = TrainingPipelineConfig()
        self.training_pipeline_config = training_pipeline_config
        self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")

            return data_ingestion_artifact
        except Exception as e:
            raise ChurnException(e,sys)
        
    def start_data_validaton(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except  Exception as e:
            raise  ChurnException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config,data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            raise ChurnException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            raise ChurnException(e,sys)
        
    def start_model_evaluation(self,model_trainer_artifact:ModelTrainerArtifact,                            
                                  data_validation_artifact:DataValidationArtifact):
        try:
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_evaluation = ModelEvaluation(data_validation_artifact=data_validation_artifact,model_eval_config=model_evaluation_config,model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact
        except Exception as e:
            raise ChurnException(e,sys)
        
    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_eval_artifact=model_eval_artifact,model_pusher_config=model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            return model_pusher_artifact
        except Exception as e:
            raise ChurnException(e,sys)

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact:DataIngestionArtifact =self.start_data_ingestion()
            data_validation_artifact = self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,data_validation_artifact=data_validation_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("trained model is not better than the best model")
            model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
            
            TrainPipeline.is_pipeline_running = False
        except Exception as e:
            raise ChurnException(e,sys)
