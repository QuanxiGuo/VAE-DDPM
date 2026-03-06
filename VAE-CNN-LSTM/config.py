

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


IC_CURVES_PATH = r"C:\Users\192052\PycharmProjects"

INPUT_SEQUENCE_LENGTH = 100
PREDICT_START_CYCLE = 300
PREDICT_END_CYCLE = 40
DQDV_FEATURE_DIM = 60

VAE_LATENT_DIM = 16
VAE_EPOCHS = 500
VAE_BATCH_SIZE = 32
VAE_LEARNING_RATE = 0.0001

CNN_LSTM_EPOCHS = 1000
CNN_LSTM_BATCH_SIZE = 16
CNN_LSTM_LEARNING_RATE = 0.001

MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
VAE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'vae_model.pth')
CNN_LSTM_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'cnn_lstm_model.pth')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)