import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_absolute_error

# Define the inverse CNN model
class InverseCDSpectrumCNN(nn.Module):
    def __init__(self):
        super(InverseCDSpectrumCNN, self).__init__()
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

        self.fc1 = nn.Linear(1024 * 30, 2048)  # Adjust input size dynamically based on the output of conv layers
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2048, 4)

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

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

# Load data
train_data = pd.read_csv('../data/cd_train_0603.csv')
test_data = pd.read_csv('../data/cd_test_0603.csv')

# Features for X and y
feature_Y = [str(i) for i in range(120)]
feature_X = ['pitch', 'fiber_radius', 'n_turns', 'helix_radius']

# Preprocess data
X_train = train_data[feature_Y].values
y_train = train_data[feature_X].values

X_test = test_data[feature_Y].values
y_test = test_data[feature_X].values

# Apply Savitzky-Golay filter to smooth the data
window_length = 15  # 滤波器窗口长度
polyorder = 3       # 多项式拟合阶数

X_train = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=X_train)
X_test = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=X_test)

# Standardize the data
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)
y_train_tensor = torch.FloatTensor(y_train)

X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test)

# Define datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Instantiate the model
model = InverseCDSpectrumCNN()
model.load_state_dict(torch.load('best_inverse_model.pth'))

import shap

test_data = np.random.rand(10, 1, 120).astype(np.float32)

# Convert the test data to a torch tensor
test_data_tensor = torch.tensor(test_data)

# Use a subset of your training data as background data for SHAP
background_data = test_data_tensor[:5]  # Example, use the first 5 samples as background

# Initialize SHAP's DeepExplainer for the PyTorch model
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values for the test data
shap_values = explainer.shap_values(test_data_tensor)

# Inspect the shape of the SHAP values
print("Shape of SHAP values:", np.array(shap_values).shape)

# Reshape test_data for SHAP summary plot (from (samples, 1, 120) to (samples, 120))
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

# Ensure SHAP values are correctly shaped for summary plot
# Depending on how the SHAP values are returned, reshape them if necessary
if len(shap_values[0].shape) > 2:
    # Assuming shap_values are in the form (samples, 1, 120)
    shap_values_reshaped = shap_values[0].reshape(shap_values[0].shape[0], -1)
else:
    shap_values_reshaped = shap_values[0]  # Use directly if the shape is already correct

shap_values_pitch = shap_values[0]  # Index 0 corresponds to the first output (pitch)
shap_values_fiber_radius = shap_values[1]
shap_values_helix_radius = shap_values[3]


# Reshape SHAP values and test data to remove the single channel dimension
# shap_values_pitch has the shape (10, 1, 120), so we reshape it to (10, 120)
shap_values_pitch_reshaped = shap_values_pitch.reshape(shap_values_pitch.shape[0], -1)

# Similarly, reshape the test data from (10, 1, 120) to (10, 120)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

shap.summary_plot(shap_values_pitch_reshaped, test_data_reshaped, feature_names=[f"Point {i+1}" for i in range(120)])


shap_values_fiber_radius_reshaped = shap_values_fiber_radius.reshape(shap_values_fiber_radius.shape[0], -1)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

shap.summary_plot(shap_values_fiber_radius_reshaped, test_data_reshaped, feature_names=[f"Point {i+1}" for i in range(120)])


shap_values_helix_radius_reshaped = shap_values_helix_radius.reshape(shap_values_helix_radius.shape[0], -1)
test_data_reshaped = test_data.reshape(test_data.shape[0], -1)

# Now plot the feature importance for pitch using the reshaped SHAP values and test data
shap.summary_plot(shap_values_helix_radius_reshaped, test_data_reshaped, feature_names=[f"Point {i+1}" for i in range(120)])


# Now plot the feature importance using the reshaped data
#shap.summary_plot(shap_values_reshaped, test_data_reshaped)
