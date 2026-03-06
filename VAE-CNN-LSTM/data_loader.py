

import pandas as pd
import numpy as np
import os
import torch  # 确保这里导入了 torch
from torch.utils.data import Dataset, DataLoader
from utils import Scaler
import config


class CycleDataset(Dataset):
    def __init__(self, data, target_data=None):
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data

        if target_data is not None:
            if isinstance(target_data, np.ndarray):
                self.target_data = torch.tensor(target_data, dtype=torch.float32)
            else:
                self.target_data = target_data
        else:
            self.target_data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.target_data is not None:
            return self.data[idx], self.target_data[idx]
        return self.data[idx]


def load_and_preprocess_dqdv(file_path, dqdv_feature_dim=config.DQDV_FEATURE_DIM):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names

    all_dqdv_curves = []
    cycle_numbers = []

    valid_sheets = []
    for sheet_name in sheet_names:
        if sheet_name.startswith('Cycle_'):
            try:
                cycle_num = int(sheet_name.split('_')[1])
                valid_sheets.append((cycle_num, sheet_name))
            except ValueError:
                continue
    valid_sheets.sort(key=lambda x: x[0])

    print(f"Loading and processing {len(valid_sheets)} dQ/dV sheets...")
    for cycle_num, sheet_name in valid_sheets:
        try:
            df_cycle = pd.read_excel(file_path, sheet_name=sheet_name)
            df_cycle.columns = df_cycle.columns.str.lower()

            if 'voltage (v)' not in df_cycle.columns or 'capacity_increment (dq/dv)' not in df_cycle.columns:
                print(f"Warning: Sheet '{sheet_name}' missing required columns. Skipping.")
                continue

            dqdv_data = df_cycle['capacity_increment (dq/dv)'].values

            if len(dqdv_data) != dqdv_feature_dim:
                print(
                    f"Warning: dQ/dV curve length for {sheet_name} is {len(dqdv_data)}, expected {dqdv_feature_dim}. Resampling.")
                original_indices = np.linspace(0, len(dqdv_data) - 1, len(dqdv_data))
                target_indices = np.linspace(0, len(dqdv_data) - 1, dqdv_feature_dim)
                dqdv_data_resampled = np.interp(target_indices, original_indices, dqdv_data)
                dqdv_data = dqdv_data_resampled

            all_dqdv_curves.append(dqdv_data)
            cycle_numbers.append(cycle_num)

        except Exception as e:
            print(f"Error reading sheet '{sheet_name}': {e}")
            continue

    if not all_dqdv_curves:
        raise ValueError("No valid dQ/dV curves loaded. Check file path and sheet names.")

    all_dqdv_curves = np.array(all_dqdv_curves)
    cycle_numbers = np.array(cycle_numbers)

    sorted_indices = np.argsort(cycle_numbers)
    return all_dqdv_curves[sorted_indices], cycle_numbers[sorted_indices]


def prepare_vae_data(dqdv_curves, scaler):
    scaler.fit(dqdv_curves)
    dqdv_scaled = scaler.transform(dqdv_curves)
    return dqdv_scaled


def prepare_cnn_lstm_data(latent_vectors, cycle_numbers,
                          input_seq_len=config.INPUT_SEQUENCE_LENGTH,
                          predict_start_cycle=config.PREDICT_START_CYCLE,
                          predict_end_cycle=config.PREDICT_END_CYCLE):

    prediction_length = predict_end_cycle - predict_start_cycle + 1

    X_train, Y_train = [], []

    predict_start_idx = np.where(cycle_numbers == predict_start_cycle)[0]
    if len(predict_start_idx) == 0:
        raise ValueError(f"Predict start cycle {predict_start_cycle} not found in loaded data.")
    predict_start_idx = predict_start_idx[0]

    max_train_end_idx = predict_start_idx - 1

    for i in range(len(latent_vectors) - input_seq_len - prediction_length + 1):
        current_input_end_idx = i + input_seq_len - 1
        current_output_end_idx = i + input_seq_len + prediction_length - 1

        if current_output_end_idx < predict_start_idx:
            X_train.append(latent_vectors[i: i + input_seq_len])
            Y_train.append(latent_vectors[i + input_seq_len: i + input_seq_len + prediction_length])
        else:
            break

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    if len(X_train) == 0:
        print(
            "Warning: CNN-LSTM training set is empty. This might happen if not enough historical data is available before the prediction start cycle.")
        print(f"Available cycles range from {cycle_numbers[0]} to {cycle_numbers[-1]}.")
        print(f"Prediction range: {predict_start_cycle}-{predict_end_cycle}")
        print(f"Required input sequence length: {input_seq_len}")
    input_vector_indices_for_prediction = np.where(cycle_numbers < predict_start_cycle)[0]

    if len(input_vector_indices_for_prediction) < input_seq_len:
        raise ValueError(
            f"Not enough historical cycles before {predict_start_cycle} for input sequence length {input_seq_len} "
            f"for the final prediction. Found {len(input_vector_indices_for_prediction)} cycles.")

    X_input_for_prediction = latent_vectors[input_vector_indices_for_prediction[-input_seq_len:]]

    predict_end_idx = np.where(cycle_numbers == predict_end_cycle)[0]
    if len(predict_end_idx) == 0:
        raise ValueError(f"Predict end cycle {predict_end_cycle} not found in loaded data.")
    predict_end_idx = predict_end_idx[0]

    Y_target_for_prediction = latent_vectors[predict_start_idx: predict_end_idx + 1]

    train_dataset = CycleDataset(X_train, Y_train)

    predict_input_dataset = CycleDataset(X_input_for_prediction[np.newaxis, :, :])

    print(f"CNN-LSTM Training Data Shape (X_train): {X_train.shape}")
    print(f"CNN-LSTM Training Data Shape (Y_train): {Y_train.shape}")
    print(f"CNN-LSTM Prediction Input Shape: {X_input_for_prediction.shape}")
    print(f"CNN-LSTM Prediction Target Shape: {Y_target_for_prediction.shape}")

    return train_dataset, predict_input_dataset, Y_target_for_prediction