# cnn_lstm_model.py

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, kernel_size=3):
        super(CNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=kernel_size,
                                padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)


        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # CNN
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_time_step_output = lstm_out[:, -1, :]
        output = self.fc(last_time_step_output)
        return output