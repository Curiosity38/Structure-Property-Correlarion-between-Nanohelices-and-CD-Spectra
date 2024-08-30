import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.signal import savgol_filter

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
window_length = 15  # Filter window length
polyorder = 3  # Polynomial order

X_train = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=X_train)
X_test = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=X_test)

# Standardize the data
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


# Define the complex DNN model
class ComplexDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.fc6(x)
        return x


# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)

# Define dataset and dataloaders
train_loader = DataLoader(TensorDataset(X_train_tensor, torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, torch.FloatTensor(y_test)), batch_size=32)

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = ComplexDNN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

# Evaluate the model
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

predictions = np.concatenate(predictions)

# Calculate R2 score, Mean Squared Error, and Mean Absolute Percentage Error for each target variable
r2_scores = []
mse_scores = []
mape_scores = []

for i, feature in enumerate(feature_X):
    r2 = r2_score(y_test[:, i], predictions[:, i])
    mse = mean_squared_error(y_test[:, i], predictions[:, i])
    mape = np.mean(np.abs((y_test[:, i] - predictions[:, i]) / y_test[:, i])) * 100
    r2_scores.append(r2)
    mse_scores.append(mse)
    mape_scores.append(mape)
    print(f"R2 score for {feature}: {r2:.4f}")
    print(f"Mean Squared Error for {feature}: {mse:.4f}")
    print(f"Mean Absolute Percentage Error for {feature}: {mape:.2f}%")

# Optionally, inverse transform the predictions and targets if they were standardized
y_test_original = y_test
predictions_original = predictions

# Plotting the results (example for one target variable)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test_original[:, 0], label='Actual')
plt.plot(predictions_original[:, 0], label='Predicted')
plt.legend()
plt.show()
