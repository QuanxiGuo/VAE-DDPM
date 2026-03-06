
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

import config
from utils import Scaler, set_seed
from data_loader import load_and_preprocess_dqdv, prepare_vae_data, prepare_cnn_lstm_data, CycleDataset
from vae_model import VAE, vae_loss_function
from cnn_lstm_model import CNNLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(6)

print("--- Loading and Preprocessing dQ/dV Data ---")
try:
    all_dqdv_curves, cycle_numbers = load_and_preprocess_dqdv(config.IC_CURVES_PATH, config.DQDV_FEATURE_DIM)
    print(
        f"Loaded {len(all_dqdv_curves)} dQ/dV curves. Max cycle: {np.max(cycle_numbers)}, Min cycle: {np.min(cycle_numbers)}")
except ValueError as e:
    print(f"Error: {e}")
    print("Please check DQDV_FEATURE_DIM in config.py and your Excel file data format, or if enough cycles exist.")
    exit()

predict_start_index = np.where(cycle_numbers == config.PREDICT_START_CYCLE)[0][0]
historical_dqdv_curves = all_dqdv_curves[:predict_start_index]
future_dqdv_curves = all_dqdv_curves[predict_start_index:]

dqdv_scaler = Scaler()
dqdv_scaler.fit(historical_dqdv_curves)

dqdv_scaled_historical = dqdv_scaler.transform(historical_dqdv_curves)
dqdv_scaled_all = dqdv_scaler.transform(all_dqdv_curves)
print(f"Normalized historical dQ/dV data shape: {dqdv_scaled_historical.shape}")
print(f"Normalized all dQ/dV data shape: {dqdv_scaled_all.shape}")

print("\n--- Training VAE for Dimensionality Reduction ---")
vae = VAE(input_dim=config.DQDV_FEATURE_DIM, latent_dim=config.VAE_LATENT_DIM).to(device)
optimizer_vae = optim.Adam(vae.parameters(), lr=config.VAE_LEARNING_RATE)

vae_dataset = CycleDataset(dqdv_scaled_historical)
vae_dataloader = DataLoader(vae_dataset, batch_size=config.VAE_BATCH_SIZE, shuffle=True)

if os.path.exists(config.VAE_MODEL_PATH):
    print(f"Loading pre-trained VAE model from {config.VAE_MODEL_PATH}")
    vae.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=device))
    vae.eval()
else:
    print(f"Training VAE for {config.VAE_EPOCHS} epochs...")
    vae.train()
    for epoch in range(config.VAE_EPOCHS):
        total_loss = 0
        for batch_idx, data in enumerate(vae_dataloader):
            data = data.to(device)
            optimizer_vae.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer_vae.step()

        avg_loss = total_loss / len(vae_dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"VAE Epoch [{epoch + 1}/{config.VAE_EPOCHS}], Loss: {avg_loss:.4f}")

    torch.save(vae.state_dict(), config.VAE_MODEL_PATH)
    print(f"VAE model saved to {config.VAE_MODEL_PATH}")
    vae.eval()

print("Reducing dimensionality with VAE...")
vae_latent_vectors = []
vae_dataset_all = CycleDataset(dqdv_scaled_all)
with torch.no_grad():
    for data in DataLoader(vae_dataset_all, batch_size=config.VAE_BATCH_SIZE, shuffle=False):
        data = data.to(device)
        mu, _ = vae.encoder(data)
        vae_latent_vectors.append(mu.cpu().numpy())
vae_latent_vectors = np.concatenate(vae_latent_vectors, axis=0)
print(f"VAE latent vectors shape: {vae_latent_vectors.shape}")

print("\n--- Preparing CNN-LSTM Data ---")

try:
    cnn_lstm_train_dataset, cnn_lstm_predict_input_dataset, cnn_lstm_predict_target_latent = \
        prepare_cnn_lstm_data(vae_latent_vectors, cycle_numbers,
                              config.INPUT_SEQUENCE_LENGTH,
                              config.PREDICT_START_CYCLE,
                              config.PREDICT_END_CYCLE)
except ValueError as e:
    print(f"Error preparing CNN-LSTM data: {e}")
    print(
        "Please check your data range, INPUT_SEQUENCE_LENGTH, PREDICT_START_CYCLE, and PREDICT_END_CYCLE in config.py.")
    exit()

if len(cnn_lstm_train_dataset) == 0:
    print("Error: CNN-LSTM training dataset is empty. Cannot train the model.")
    print(
        "This might happen if not enough historical data is available before the prediction start cycle, or if filtering conditions are too strict.")
    print(f"Loaded cycles range: {cycle_numbers[0]} to {cycle_numbers[-1]}")
    exit()

cnn_lstm_train_dataloader = DataLoader(cnn_lstm_train_dataset, batch_size=config.CNN_LSTM_BATCH_SIZE, shuffle=True)
cnn_lstm_predict_dataloader = DataLoader(cnn_lstm_predict_input_dataset, batch_size=1, shuffle=False)

print("\n--- Training CNN-LSTM Model ---")
prediction_length_latent = config.PREDICT_END_CYCLE - config.PREDICT_START_CYCLE + 1
cnn_lstm = CNNLSTM(
    input_dim=config.VAE_LATENT_DIM,
    hidden_dim=64,
    num_layers=2,
    output_dim=prediction_length_latent * config.VAE_LATENT_DIM
).to(device)
optimizer_cnn_lstm = optim.Adam(cnn_lstm.parameters(), lr=config.CNN_LSTM_LEARNING_RATE)
criterion_cnn_lstm = nn.MSELoss()

if os.path.exists(config.CNN_LSTM_MODEL_PATH):
    print(f"Loading pre-trained CNN-LSTM model from {config.CNN_LSTM_MODEL_PATH}")
    cnn_lstm.load_state_dict(torch.load(config.CNN_LSTM_MODEL_PATH, map_location=device))
    cnn_lstm.eval()
else:
    print(f"Training CNN-LSTM for {config.CNN_LSTM_EPOCHS} epochs...")
    cnn_lstm.train()
    for epoch in range(config.CNN_LSTM_EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(cnn_lstm_train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets_flat = targets.view(targets.size(0), -1)
            optimizer_cnn_lstm.zero_grad()
            outputs = cnn_lstm(inputs)
            loss = criterion_cnn_lstm(outputs, targets_flat)
            loss.backward()
            total_loss += loss.item()
            optimizer_cnn_lstm.step()
        avg_loss = total_loss / len(cnn_lstm_train_dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"CNN-LSTM Epoch [{epoch + 1}/{config.CNN_LSTM_EPOCHS}], Loss: {avg_loss:.4f}")
    torch.save(cnn_lstm.state_dict(), config.CNN_LSTM_MODEL_PATH)
    print(f"CNN-LSTM model saved to {config.CNN_LSTM_MODEL_PATH}")
    cnn_lstm.eval()

print("\n--- Making Predictions ---")
predicted_latent_vectors = []
with torch.no_grad():
    for inputs in cnn_lstm_predict_dataloader:
        inputs = inputs.to(device)
        outputs = cnn_lstm(inputs)
        predicted_latent_vectors.append(outputs.view(prediction_length_latent, config.VAE_LATENT_DIM).cpu().numpy())

predicted_latent_vectors = np.concatenate(predicted_latent_vectors, axis=0)
print(f"Predicted latent vectors shape: {predicted_latent_vectors.shape}")

predicted_dqdv_scaled = vae.decoder(
    torch.tensor(predicted_latent_vectors, dtype=torch.float32).to(device)).cpu().detach().numpy()
predicted_dqdv_curves = dqdv_scaler.inverse_transform(predicted_dqdv_scaled)
print(f"Predicted dQ/dV curves shape: {predicted_dqdv_curves.shape}")

print("\n--- Evaluating and Visualizing Results ---")

true_target_dqdv_scaled = vae.decoder(
    torch.tensor(cnn_lstm_predict_target_latent, dtype=torch.float32).to(device)).cpu().detach().numpy()
true_target_dqdv_curves = dqdv_scaler.inverse_transform(true_target_dqdv_scaled)

sample_voltage_data = pd.read_excel(config.IC_CURVES_PATH, sheet_name=f'Cycle_{cycle_numbers[0]}')
sample_voltage_data.columns = sample_voltage_data.columns.str.lower()
voltage_ref_original = sample_voltage_data['voltage (v)'].values

original_indices_voltage = np.linspace(0, len(voltage_ref_original) - 1, len(voltage_ref_original))
target_indices_voltage = np.linspace(0, len(voltage_ref_original) - 1, config.DQDV_FEATURE_DIM)
voltage_ref = np.interp(target_indices_voltage, original_indices_voltage, voltage_ref_original)
print(f"Resampled voltage_ref shape: {voltage_ref.shape}. Expected {config.DQDV_FEATURE_DIM}.")


def calculate_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def calculate_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))


output_dir = os.path.join(config.BASE_DIR, 'prediction_plots')
os.makedirs(output_dir, exist_ok=True)
print(f"Saving individual plots to: {output_dir}")

excel_output_path = os.path.join(config.BASE_DIR, '')
writer = pd.ExcelWriter(excel_output_path, engine='xlsxwriter')
print(f"Saving prediction data to Excel file: {excel_output_path}")

all_metrics = []

for i in range(prediction_length_latent):
    cycle_offset = config.PREDICT_START_CYCLE + i

    true_dqdv = true_target_dqdv_curves[i]
    predicted_dqdv = predicted_dqdv_curves[i]

    rmse_value = calculate_rmse(predicted_dqdv, true_dqdv)
    mae_value = calculate_mae(predicted_dqdv, true_dqdv)

    all_metrics.append({
        'Cycle': cycle_offset,
        'RMSE': rmse_value,
        'MAE': mae_value
    })

    df_output = pd.DataFrame({
        'Voltage (V)': voltage_ref,
        'True dQ/dV': true_dqdv,
        'Predicted dQ/dV': predicted_dqdv
    })
    df_output.to_excel(writer, sheet_name=f'Cycle_{cycle_offset}', index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(voltage_ref, true_dqdv, label=f'True Cycle {cycle_offset}', color='blue', linewidth=2)
    plt.plot(voltage_ref, predicted_dqdv, label=f'Predicted Cycle {cycle_offset}', color='red', linestyle='--',
             linewidth=2)
    plt.title(f'Cycle {cycle_offset} dQ/dV Prediction')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Capacity Increment (dQ/dV)')
    plt.text(0.05, 0.95,
             f'RMSE: {rmse_value:.4f}\nMAE: {mae_value:.4f}',
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show(block=False)

    plot_filename = os.path.join(output_dir,
                                 f'Cycle_{cycle_offset}_dQdV_Prediction_RMSE_{rmse_value:.4f}_MAE_{mae_value:.4f}.png')
    plt.close()

df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

writer.close()
print(f"\nPrediction data successfully saved to {excel_output_path}")
print(f"\nIndividual plots with RMSE and MAE saved to {output_dir}")
print("\nPrediction and visualization complete.")