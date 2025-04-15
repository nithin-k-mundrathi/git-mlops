import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    
    def __init__(self, train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir =  processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocessed_data(self,df):
        try:
            logger.info('Starting Our Data Processing step')
            logger.info("dropping columns")

            df.drop(columns = ["Unnamed: 0","Booking_ID"],inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols =  self.config["data_processing"]['categorical_columns']
            num_cols =  self.config["data_processing"]['numerical_columns']

            logger.info('Apply label Encoding')

            label_encoder = preprocessing.LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for label, code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}

            logger.info('label mappings are :')
            # for col,mapping in mappings.items():
            #     logger.info(f"{col} : {mapping}")

            logger.info("Doing skewness")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].apply(lambda x:x.skew())

            for col in skewness[skewness> skew_threshold].index:
                df[col] = np.log1p(df[col])
            return df

        except Exception as e:
            logger.error(f"Error During pre_process step {e}")
            raise CustomException("error during preprocess data", e)

    def balance_data(self, df):
        try:
            logger.info('Handling Imbalanced data')
            y = df['booking_status']
            X = df.drop('booking_status',axis=1)
        
            smote = SMOTE(random_state=7)

            X_ressampled , y_resampled = smote.fit_resample(X,y)
            df_balanced = pd.DataFrame(X_ressampled,columns = X.columns)
            df_balanced['booking_status'] = y_resampled

            logger.info("Data balancing Successful")

            return df_balanced
        except Exception as e:
            logger.error(f"Error during balancing step {e}")
            raise CustomException("Error while preprocessing data", e)
    
    def select_features(self,df):
        try:
            logger.info("Starting our feature selection step")

            y = df['booking_status']
            X = df.drop('booking_status',axis=1)
            

            model = RandomForestClassifier(random_state=8)
            model.fit(X,y)

            feature_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame(
                {
                    'feature' : X.columns,
                    'importance' : feature_importance
                }
            )

            top_feature_importance_df = feature_importance_df.sort_values(
                by = "importance", ascending=False
            )

            num_features_to_select = self.config["data_processing"]["no_of_features"]
            top_10_features = top_feature_importance_df['feature'].head(num_features_to_select).values
            
            logger.info(f'features selected',top_10_features)
            top_10_df =  df[top_10_features.tolist() + ["booking_status"]]

            logger.info('feature Selection complted successfully')

            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection {e}")
            raise CustomException("Error while feature selection data", e)
    
    
    def save_data(self,df,file_path):
        try:

            logger.info('saving our data into processed folder')

            df.to_csv(file_path,index=False)

            logger.info(f"data Saved successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error during Saving Data {e}")
            raise CustomException("Error while Saving data", e)
        
    def process(self):
        try:
            
            logger.info('Loading data from RAW directory')

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocessed_data(train_df)
            test_df = self.preprocessed_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df =  test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data Processing Completed Successfully")

        except Exception as e:
            logger.error(f"Error during processing Data {e}")
            raise CustomException("Error while processing Data", e)
        

if __name__ == "__main__":
    processor = DataProcessor(
        TRAIN_FILE_PATH,
        TEST_FILE_PATH,
        PROCESSED_DIR,
        CONFIG_PATH
    )

    processor.process()