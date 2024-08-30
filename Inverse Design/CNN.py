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

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
test_losses = []

# Define lists to store train and test R2 scores
train_r2_scores = []
test_r2_scores = []
best_test_r2 = -float('inf')
best_epoch = -1

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    test_loss = 0
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions)

    # Calculate test loss and R2 score
    test_loss /= len(test_loader.dataset)
    test_r2 = r2_score(y_test, predictions)
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_inverse_model.pth')

    # Append values to lists
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_r2_scores.append(test_r2)
    if epoch % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test R2: {test_r2}")

print(f"Best R2 score achieved at epoch {best_epoch + 1}")
print(f"Best test R2 score {best_test_r2}")

# Load the best model
model.load_state_dict(torch.load('best_inverse_model.pth'))
model.eval()

# Predict on the test set
test_loss = 0
predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

predictions = np.concatenate(predictions)

# Inverse transform the predictions and ground truth values
y_test = scaler_y.inverse_transform(y_test)
predictions = scaler_y.inverse_transform(predictions)

# Calculate final R2 score
final_r2 = r2_score(y_test, predictions)
print("Final R2 score:", final_r2)

# Calculate and print percentage prediction error for each parameter
absolute_errors = np.abs(predictions - y_test)
percentage_errors = absolute_errors / np.abs(y_test) * 100

# Save predictions, actual values, and errors to a CSV file
output_df = pd.DataFrame(predictions, columns=[f'pred_{x}' for x in feature_X])
actual_df = pd.DataFrame(y_test, columns=[f'actual_{x}' for x in feature_X])
error_df = pd.DataFrame(percentage_errors, columns=[f'error_{x}' for x in feature_X])

result_df = pd.concat([actual_df, output_df, error_df], axis=1)
result_df.to_csv('prediction_results.csv', index=False)

# Print out some results
print(result_df.head())

# Plotting losses
sns.set(style="white", palette="Set2")
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='red')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Losses')
plt.legend()
plt.show()
