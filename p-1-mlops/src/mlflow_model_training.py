import os
import pandas as pd # type: ignore
import joblib # type: ignore
from sklearn.model_selection import RandomizedSearchCV
from src.custom_exception import CustomException
from lightgbm import LGBMClassifier # type: ignore # type: ignore
from sklearn import metrics
from src.logger import get_logger
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
import mlflow

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

logger =  get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist =  LIGHTGBM_params
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):

        try:
            logger.info(f'Loading Data from {self.train_path}')
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(['booking_status'],axis=1)
            y_train = train_df['booking_status']

            X_test = test_df.drop(['booking_status'],axis=1)
            y_test = test_df['booking_status']

            logger.info("data splitted successfully for Model Training")

            return X_train,y_train,X_test,y_test

        except Exception as e:
            logger.error(f"Error during Loading Data in Model training{e}")
            raise CustomException("Error while Loading Data in Model Training", e)
        
    
    def train_lgbm(self,X_train,y_train):
        try:
            logger.info("Initializing Our Model")
            lgbm_model = LGBMClassifier(
                random_state = self.random_search_params['random_state']
            )

            logger.info("Starting our hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator = lgbm_model,
                param_distributions = self.params_dist,
                n_iter = self.random_search_params['n_iter'],
                cv = self.random_search_params['cv'],
                n_jobs = self.random_search_params['n_jobs'],
                verbose = self.random_search_params['verbose'],
                random_state = self.random_search_params['random_state'],
                scoring = self.random_search_params['scoring']
            )

            logger.info('Starting our model Training')
            random_search.fit(X_train, y_train)

            logger.info('Hyperparameter training')

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f'best parameters are : {best_params}')

            return best_lgbm_model

        except Exception as e:
            logger.error(f"Error during hyperparameterization in Model training{e}")
            raise CustomException("Error while hyperparameterization in Model Training", e)
    
    def evaluate_model(self,model,X_test,y_test):
        try:
            logger.info('Evaluate the Model')

            y_pred = model.predict(X_test)

            accuracy = metrics.accuracy_score(y_test,y_pred)

            logger.info(f"Accuracy score; {accuracy}")

            return {
                 'accuracy': accuracy 
            }
        
        except Exception as e:
            logger.error(f"Error during Evaluate Model in Model training{e}")
            raise CustomException("Error while Evaluate Model in Model Training", e)


    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok = True)

            logger.info("saving the model")

            joblib.dump(model, self.model_output_path)
            logger.info(f"model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error during Saving Model in Model training{e}")
            raise CustomException("Error while saving Model in Model Training", e)

    def run(self):

        try:
            with mlflow.start_run():
                logger.info(f"start our model Training Pipeline")
                logger.info('Starting our MLFLOW experimentation')

                logger.info('Logging the Training and testing Dataset to MLFLOW')
                mlflow.log_artifact(self.train_path, artifact_path = 'datasets')
                mlflow.log_artifact(self.test_path, artifact_path = 'datasets')
                
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                model_metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                logger.info("Loging Model to MLFLOW")
                # mlflow.log_artifacts(self.model_output_path)

                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(model_metrics)

                logger.info('model Training Successfully Completed')

        except Exception as e:
            logger.error(f"Error during RUN in Model training{e}")
            raise CustomException("Error while RUN in Model Training", e)
        
if __name__ == "__main__":
    trainer =  ModelTraining(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    trainer.run()

