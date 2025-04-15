import os

########################  DATA INGESTION ############################
RAW_DIR = "artifacts/raw"
CONFIG_PATH = "config/config.yaml"

########################  DATA PROCESSING ############################
PROCESSED_DIR = "artifacts/raw"
ANIME_LIST_CSV = os.path.join(RAW_DIR,"animelist.csv")
ANIME_CSV = os.path.join(RAW_DIR,"anime.csv")
ANIME_SYNOPSIS_CSV = os.path.join(RAW_DIR,"anime_with_synopsis.csv")

X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR,"X_train_array.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR,"X_test_array.pkl")
Y_TRAIN = os.path.join(PROCESSED_DIR,"y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR,"y_test.pkl")

RATING_DF = os.path.join(PROCESSED_DIR,"rating_df.csv") 
DF =  os.path.join(PROCESSED_DIR,"anime_df.csv")
SYNOPSIS_DF = os.path.join(PROCESSED_DIR,"synopsis_df.csv")

USER2USER_ENCODED  = os.path.join(PROCESSED_DIR,"user2user_encoded.pkl")
USER2USER_DECODED  = os.path.join(PROCESSED_DIR,"user2user_decoded.pkl")

ANIME2ANIME_ENCODED = os.path.join(PROCESSED_DIR,"anime2anime_encoded.pkl")
ANIME2ANIME_DECODED = os.path.join(PROCESSED_DIR,"anime2anime_decoded.pkl")

########################  MODEL TRAINING ############################

MODEL_DIR = 'artifacts/model'
WEIGHTS_DIR = 'artifacts/weights'
MODEL_PATH = os.path.join(MODEL_DIR,'model.h5')
ANIME_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR,'anime_weights.pkl')
USER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR,'user_weights.pkl')
CHECKPOINT_FILE_PATH = 'artifacts/model_checkpoint/weights.weights.h5'

