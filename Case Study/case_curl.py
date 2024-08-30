import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error
import math

# Define the CNN model
class CDSpectrumCNN(nn.Module):
    def __init__(self):
        super(CDSpectrumCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.bn2 = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 2048)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2048, 120)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x

# Load and preprocess data
train_data = pd.read_csv('data/cd_train_0603.csv')
test_data = pd.read_csv('data/cd_test_0603.csv')

# Define feature columns
feature_Y = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
             '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54',
             '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
             '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90',
             '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107',
             '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119']

X_train = train_data[['pitch', 'fiber_radius', 'n_turns', 'helix_radius']].values
y_train = train_data[feature_Y].values

X_test = test_data[['pitch', 'fiber_radius', 'n_turns', 'helix_radius']].values
y_test = test_data[feature_Y].values

window_length = 15
polyorder = 3

# Apply Savitzky-Golay filter
y_train = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=y_train)
y_test = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=y_test)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train * 1e15)
y_test = y_scaler.transform(y_test * 1e15)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Define datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
test_dataset = TensorDataset(X_test_tensor.unsqueeze(1), y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Instantiate the model
model = CDSpectrumCNN()
# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Load the best model
model.load_state_dict(torch.load('best_model_mae_5.pth'))
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the fixed parameters
fixed_parameters = {
    'pitch': 100,
    'fiber_radius': 30,
    'n_turns': 4,
}

# Define curl range and calculate corresponding helix_radius values
curl_range = np.linspace(4, 8, 50)
helix_radius_range = curl_range * fixed_parameters['pitch'] / (2 * np.pi)

# Prepare input data for prediction
def generate_input_data(fixed_params, helix_radius_range):
    input_data = []
    for helix_radius in helix_radius_range:
        params = fixed_params.copy()
        params['helix_radius'] = helix_radius
        input_data.append([params['pitch'], params['fiber_radius'], params['n_turns'], params['helix_radius']])
    return np.array(input_data)

X_varied = generate_input_data(fixed_parameters, helix_radius_range)

# Standardize the varied input data
X_varied = scaler.transform(X_varied)

# Convert to PyTorch tensor
X_varied_tensor = torch.FloatTensor(X_varied).unsqueeze(1).to(device)

# Predict using the trained model
model.eval()
with torch.no_grad():
    predictions_varied = model(X_varied_tensor).cpu().numpy()

# Inverse transform the predictions
predictions_varied = y_scaler.inverse_transform(predictions_varied)
predictions_varied = predictions_varied / 1e15

# Apply the Savitzky-Golay filter to smooth the predictions
predictions_varied = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=predictions_varied)

# Plot the results with color scale based on curl
sns.set(style="white", palette="Set2")
x_sequence = np.linspace(400, 800, 120)
fig, ax = plt.subplots(figsize=(20, 12))  # Create a figure and an axis
plt.rcParams['axes.linewidth'] = 5

# Create a colormap for curl
cmap = plt.get_cmap('Paired')
norm = plt.Normalize(vmin=curl_range.min(), vmax=curl_range.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for i, prediction in enumerate(predictions_varied):
    color = cmap(norm(curl_range[i]))
    sns.lineplot(x=x_sequence, y=prediction, color=color, linestyle='-', linewidth=2, ax=ax)

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Ext,Diff')
ax.set_title('Effect of curl on CD Spectrum')

# Add colorbar with a thinner border
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Curl value')
cbar.outline.set_linewidth(1)  # Set the colorbar outline thickness to 1

plt.show()
