import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import warnings
import joblib
import os


class dQVDataloader:

    def __init__(self, file_path: str, cycle_range: range, target_length: int = 100):
        self.data = {}
        self.dqdv_curves = []
        self.cycle_numbers_for_dqdv = []
        self.target_length = target_length
        self.dqdv_scaler = MinMaxScaler()

        print(f"Loading dQ/dV data from: {file_path}")
        try:
            excel_file = pd.ExcelFile(file_path)
            all_dqdv_data_for_scaler = []
            all_voltages = []

            for cycle_num in cycle_range:
                sheet_name = f'Cycle_{cycle_num}'
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    # 灵活匹配电压列名
                    v_col = next((c for c in df.columns if 'voltage' in str(c).lower() or 'u /v' in str(c).lower()),
                                 None)
                    if v_col:
                        all_voltages.extend(df[v_col].dropna().tolist())

            min_v, max_v = (np.min(all_voltages), np.max(all_voltages)) if all_voltages else (3.0, 4.2)

            for cycle_num in cycle_range:
                sheet_name = f'Cycle_{cycle_num}'
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    v_col = next((c for c in df.columns if 'voltage' in str(c).lower()), None)
                    dq_col = next(
                        (c for c in df.columns if 'dq/dv' in str(c).lower() or 'capacity_increment' in str(c).lower()),
                        None)

                    if v_col and dq_col:
                        v, d = df[v_col].dropna().values, df[dq_col].dropna().values
                        if len(v) > 1:
                            idx = np.argsort(v)
                            uv, uid = np.unique(v[idx], return_index=True)
                            ud = d[idx][uid]
                            f = interp1d(uv, ud, kind='linear', bounds_error=False, fill_value="extrapolate")
                            resampled = f(np.linspace(min_v, max_v, self.target_length))
                            self.dqdv_curves.append(resampled)
                            self.cycle_numbers_for_dqdv.append(cycle_num)
                            all_dqdv_data_for_scaler.extend(resampled)

            if all_dqdv_data_for_scaler:
                self.dqdv_scaler.fit(np.array(all_dqdv_data_for_scaler).reshape(-1, 1))
                self.dqdv_curves = [self.dqdv_scaler.transform(c.reshape(-1, 1)).flatten() for c in self.dqdv_curves]
        except Exception as e:
            print(f"Error loading dQ/dV: {e}");
            raise

    def get_cycle_to_dqdv_map(self, vae_model, device):
        dqdv_latent_map = {}
        vae_model.eval()
        with torch.no_grad():
            for i, cycle_num in enumerate(self.cycle_numbers_for_dqdv):
                dqdv_tensor = torch.FloatTensor(self.dqdv_curves[i]).unsqueeze(0).to(device)
                _, _, _, latent_vec = vae_model(dqdv_tensor)
                dqdv_latent_map[cycle_num] = latent_vec.squeeze(0).cpu()
        return dqdv_latent_map


class BatteryDatasetConditional(Dataset):

    def __init__(self, file_path: str, dqdv_latent_map: dict, cycle_range: range, target_length: int = 850,
                 scalers_path: str = None):
        self.data = []
        self.dqdv_latent_map = dqdv_latent_map
        self.target_length = target_length
        self.scalers = {'voltage': MinMaxScaler(), 'current': MinMaxScaler(), 'soc': MinMaxScaler()}

        if scalers_path and os.path.exists(scalers_path):
            print(f"Loading pre-defined scalers from: {scalers_path}")
            self.scalers = joblib.load(scalers_path)
            self.fit_scalers = False
        else:
            self.fit_scalers = True

        try:
            excel_file = pd.ExcelFile(file_path)
            all_data_for_fit = {'voltage': [], 'current': [], 'soc': []}

            for cycle_num in cycle_range:
                sheet_name = f'Cycle_{cycle_num}'
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    v_col, i_col, s_col, t_col = self._find_columns_with_time(df)

                    if all([v_col, i_col, s_col, t_col]):
                        v, i, s, t = df[v_col].dropna().values, df[i_col].dropna().values, \
                            df[s_col].dropna().values, df[t_col].dropna().values

                        min_len = min(len(v), len(i), len(s), len(t))
                        if min_len > 0:
                            self.data.append({
                                'voltage': v[:min_len],
                                'current': i[:min_len],
                                'soc': s[:min_len],
                                'raw_time': t[:min_len],
                                'cycle': cycle_num
                            })
                            if self.fit_scalers:
                                all_data_for_fit['voltage'].extend(v[:min_len])
                                all_data_for_fit['current'].extend(i[:min_len])
                                all_data_for_fit['soc'].extend(s[:min_len])

            if self.fit_scalers and self.data:
                for k in self.scalers:
                    self.scalers[k].fit(np.array(all_data_for_fit[k]).reshape(-1, 1))

            for entry in self.data:
                for f in ['voltage', 'current', 'soc']:
                    resized = self._resize_sequence(entry[f], self.target_length)
                    entry[f'{f}_norm'] = self.scalers[f].transform(resized.reshape(-1, 1)).flatten()

        except Exception as e:
            print(f"Error loading discharge data: {e}");
            raise

    def _find_columns_with_time(self, df):
        v_c, i_c, s_c, t_c = None, None, None, None
        for col in df.columns:
            c_s = str(col).lower()
            if ('u' in c_s or 'voltage' in c_s) and 'v' in c_s:
                v_c = col
            elif ('i' in c_s and 'a' in c_s) and all(x not in c_s for x in ['temp', 'capacity']):
                i_c = col
            elif 'soc' in c_s:
                s_c = col
            elif 'time' in c_s:
                t_c = col
        return v_c, i_c, s_c, t_c

    def _resize_sequence(self, sequence, target_length):
        return interp1d(np.linspace(0, 1, len(sequence)), sequence, kind='linear')(np.linspace(0, 1, target_length))

    def save_scalers(self, path):
        joblib.dump(self.scalers, path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        combined = torch.stack([torch.FloatTensor(d[f'{f}_norm']) for f in ['voltage', 'current', 'soc']], dim=0)
        dqdv_latent = self.dqdv_latent_map.get(d['cycle'], torch.zeros(16))
        return combined, dqdv_latent, torch.LongTensor([d['cycle']])