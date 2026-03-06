
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

from vae_model import Encoder, Decoder, VAE
from diffusion_model_architecture import UNet1DConditional
from diffusion_model_logic import DiffusionModelConditional
from battery_data_processing import dQVDataloader, BatteryDatasetConditional
from utils import set_seed

MODEL_SAVE_DIR = 'models_VAE_DM'
VAE_MODEL_FILENAME = 'dqdv_vae_model.pth'
DIFFUSION_MODEL_FILENAME = 'conditional_diffusion_model.pth'

DISCHARGE_DATA_PATH = r"C:\Users"
DQDV_DATA_PATH = r"C:\Users"

DQDV_TARGET_LENGTH = 100
DQDV_VAE_LATENT_DIM = 16
CYCLE_EMB_DIM = 16
MAX_TOTAL_CYCLE_NUM = 400

DISCHARGE_TARGET_LENGTH = None

TARGET_CYCLE_NUMBERS_TO_GENERATE = list(range(2, 100))
NUM_SAMPLES_PER_CYCLE = 1

OUTPUT_EXCEL_PATH = ''
PLOT_NUM_SAMPLES = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_models(
    model_save_dir: str,
    vae_model_filename: str,
    diffusion_model_filename: str,
    dqdv_target_length: int,
    dqdv_vae_latent_dim: int,
    cycle_emb_dim: int,
    max_total_cycle_num: int,
) -> tuple[VAE, UNet1DConditional]:
    print(f"Loading models from directory: {model_save_dir}")

    # --- 加载 VAE 模型 ---
    vae_encoder = Encoder(input_dim=dqdv_target_length, latent_dim=dqdv_vae_latent_dim).to(device)
    vae_decoder = Decoder(latent_dim=dqdv_vae_latent_dim, output_dim=dqdv_target_length).to(device)
    vae_model = VAE(vae_encoder, vae_decoder).to(device)
    vae_model_path = os.path.join(model_save_dir, vae_model_filename)

    if not os.path.exists(vae_model_path):
        raise FileNotFoundError(f"VAE Model file not found at: {vae_model_path}")

    vae_model.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae_model.eval()
    print(f"Successfully loaded VAE model from {vae_model_path}")

    diffusion_model = UNet1DConditional(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        dqdv_latent_dim=dqdv_vae_latent_dim,
        max_cycle_num=max_total_cycle_num,
        cycle_emb_dim=cycle_emb_dim
    ).to(device)
    diffusion_model_path = os.path.join(model_save_dir, diffusion_model_filename)

    if not os.path.exists(diffusion_model_path):
        raise FileNotFoundError(f"Diffusion Model file not found at: {diffusion_model_path}")

    diffusion_model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    diffusion_model.eval()
    print(f"Successfully loaded Conditional Diffusion Model from {diffusion_model_path}")

    return vae_model, diffusion_model

def generate_synthetic_data(
    unet_model: UNet1DConditional,
    diffusion_logic: DiffusionModelConditional,
    dataset_obj: BatteryDatasetConditional,
    dqdv_latent_map: dict,
    target_cycle_numbers: list,
    num_samples_per_cycle: int = 1,
    device=None
) -> list[dict]:

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet_model.eval()
    synthetic_cycles = []

    seq_length = dataset_obj.target_length
    if seq_length is None:
        raise ValueError("BatteryDatasetConditional's target_length is None. It must be determined before generation.")

    num_features = 3

    print(f"\n--- Generating Synthetic Data for cycles: {target_cycle_numbers} ---")
    with torch.no_grad():
        for cycle_num in target_cycle_numbers:
            if cycle_num not in dqdv_latent_map:
                warnings.warn(f"dQ/dV latent vector not found for target cycle {cycle_num}. Skipping generation for this cycle.")
                continue

            dqdv_latent = dqdv_latent_map[cycle_num].unsqueeze(0).repeat(num_samples_per_cycle, 1).to(device)
            cycle_num_tensor = torch.LongTensor([cycle_num]).repeat(num_samples_per_cycle).to(device)

            shape = (num_samples_per_cycle, num_features, seq_length)
            synthetic_data_batch = diffusion_logic.p_sample(
                shape, target_dqdv_latents=dqdv_latent, target_cycle_numbers=cycle_num_tensor
            )

            synthetic_data_batch = synthetic_data_batch.cpu().numpy()

            for i in range(num_samples_per_cycle):
                voltage_norm = synthetic_data_batch[i, 0, :]
                current_norm = synthetic_data_batch[i, 1, :]
                soc_norm = synthetic_data_batch[i, 2, :]

                voltage = dataset_obj.scalers['voltage'].inverse_transform(voltage_norm.reshape(-1, 1)).flatten()
                current = dataset_obj.scalers['current'].inverse_transform(current_norm.reshape(-1, 1)).flatten()
                soc = dataset_obj.scalers['soc'].inverse_transform(soc_norm.reshape(-1, 1)).flatten()

                synthetic_cycles.append({
                    'cycle': cycle_num,
                    'sample_idx': i,
                    'voltage': voltage,
                    'current': current,
                    'soc': soc,
                    'time': np.arange(seq_length) * 8
                })
    print(f"Generated {len(synthetic_cycles)} synthetic samples.")
    return synthetic_cycles


def plot_generated_data(synthetic_data_list: list[dict], num_plots: int = 3):

    if not synthetic_data_list:
        print("No synthetic data to plot.")
        return

    feature_names = ['Voltage (V)', 'Current (A)', 'SOC (%)']
    feature_keys = ['voltage', 'current', 'soc']
    num_features = len(feature_names)

    plots_to_show = min(num_plots, len(synthetic_data_list))

    fig, axes = plt.subplots(num_features, plots_to_show, figsize=(5 * plots_to_show, 4 * num_features), squeeze=False)
    plt.suptitle("Generated Synthetic Battery Data (V, I, SOC)", fontsize=16)

    print(f"Plotting {plots_to_show} generated cycles.")
    for i in range(plots_to_show):
        cycle_data = synthetic_data_list[i]
        for j, (feature_name, feature_key) in enumerate(zip(feature_names, feature_keys)):
            ax = axes[j, i]
            ax.plot(cycle_data['time'], cycle_data[feature_key], label=f'Synthetic Cycle {cycle_data["cycle"]}', color='red')
            ax.set_title(f'{feature_name} (Generated Cycle {cycle_data["cycle"]})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(feature_name)
            ax.legend()
            ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def save_synthetic_data_to_excel(synthetic_data_list: list[dict], output_path: str):

    if not synthetic_data_list:
        print("No synthetic data to save.")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for i, cycle_data in enumerate(synthetic_data_list):
            sheet_name = f'Cycle_{cycle_data["cycle"]}_Sample_{cycle_data.get("sample_idx", 0)}'
            df = pd.DataFrame({
                'Time (s)': cycle_data['time'],
                'U /V': cycle_data['voltage'],
                'I /A': cycle_data['current'],
                'SOC': cycle_data['soc']
            })
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Synthetic data saved to {output_path}")


if __name__ == "__main__":

    print("--- Starting Data Generation from Pre-trained Models ---")

    try:
        vae_model, unet_model = load_models(
            MODEL_SAVE_DIR,
            VAE_MODEL_FILENAME,
            DIFFUSION_MODEL_FILENAME,
            DQDV_TARGET_LENGTH,
            DQDV_VAE_LATENT_DIM,
            CYCLE_EMB_DIM,
            MAX_TOTAL_CYCLE_NUM,
            DISCHARGE_TARGET_LENGTH
        )
        diffusion_logic = DiffusionModelConditional(unet_model, device=device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your pre-trained model files are in the specified directory.")
        exit()
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        import traceback
        traceback.print_exc()
        exit()

    print("\nInitializing data loaders to retrieve scalers and dQ/dV latent map...")
    try:

        dummy_train_cycles = range(2, MAX_TOTAL_CYCLE_NUM + 1)
        dataset_obj_for_scalers = BatteryDatasetConditional(
            DISCHARGE_DATA_PATH,
            {},
            cycle_range=dummy_train_cycles,
            target_length=DISCHARGE_TARGET_LENGTH
        )

        dqdv_data_loader_obj = dQVDataloader(DQDV_DATA_PATH, cycle_range=range(2, MAX_TOTAL_CYCLE_NUM + 1), target_length=DQDV_TARGET_LENGTH)

        dqdv_latent_map_for_generation = dqdv_data_loader_obj.get_cycle_to_dqdv_map(vae_model, device)

        if not dqdv_latent_map_for_generation:
            print("Error: No dQ/dV latent vectors could be generated from the loaded VAE. Exiting.")
            exit()

    except Exception as e:
        print(f"Error initializing data loaders or generating dQ/dV latent map: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure your data file paths are correct and data processing works as expected.")
        exit()

    synthetic_data_list = generate_synthetic_data(
        unet_model,
        diffusion_logic,
        dataset_obj_for_scalers,
        dqdv_latent_map_for_generation,
        TARGET_CYCLE_NUMBERS_TO_GENERATE,
        NUM_SAMPLES_PER_CYCLE,
        device
    )

    plot_generated_data(synthetic_data_list, num_plots=PLOT_NUM_SAMPLES)

    save_synthetic_data_to_excel(synthetic_data_list, OUTPUT_EXCEL_PATH)

    print("\nData generation process completed!")