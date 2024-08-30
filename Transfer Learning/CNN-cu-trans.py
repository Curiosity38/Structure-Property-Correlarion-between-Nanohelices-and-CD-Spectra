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


# Define the CNN model
class CDSpectrumCNN(nn.Module):
    def __init__(self):
        super(CDSpectrumCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)  # 添加Dropout层
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)  # 添加Dropout层
        self.bn2 = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.bn3 = nn.BatchNorm1d(1024)

        # Dynamic calculation of the number of input features for the fully connected layer
        self.num_flat_features = None

        # Define the fully connected layers
        self.fc1 = nn.Linear(1024, 2048)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(2048, 120)
        #self.fc3 = nn.Linear(512, 256)

        #self.fc4 = nn.Linear(256, 60)


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
        x = self.pool3(x)
        x = self.dropout3(x)

        # Calculate the number of features after flattening
        if self.num_flat_features is None:
            self.num_flat_features = x.size(1) * x.size(2)

        x = x.view(-1, self.num_flat_features)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        #x = self.fc3(x)
        #x = self.fc4(x)

        return x


train_data = pd.read_csv('data/train_df_cd_cu.csv')
test_data = pd.read_csv('data/test_df_cd_cu.csv')

# Load and preprocess data
feature_Y = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20',
             '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
             '39', '40',
             '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
             '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76',
             '77',
             '78',
             '79', '80',
             '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
             '99', '100',
             '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115',
             '116', '117', '118',
             '119'

             ]
'''

'''

X_train = train_data[
    ['pitch', 'fiber_radius', 'n_turns', 'helix_radius', 'curl', 'total_length', 'angle', 'height',
     'total_fiber_length', 'V']].values
# , 'x_y', 'direction', 'LCP_RCP', 'section'
y_train = train_data[feature_Y].values

X_test = test_data[
    ['pitch', 'fiber_radius', 'n_turns', 'helix_radius', 'curl', 'total_length', 'angle', 'height',
     'total_fiber_length', 'V']].values
y_test = test_data[feature_Y].values

window_length = 39  # 滤波器窗口长度
polyorder = 2  # 多项式拟合阶数

# 对训练集数据应用滤波
y_train = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=y_train)

# 对测试集数据应用滤波
y_test = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=y_test)

#sns.set(style="darkgrid", palette="Set2")
x_sequence = np.linspace(400, 800, 120)
'''
for index in range(0, 800):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_sequence, y=y_train[index], linewidth=5)
    min_y_value = np.min(y_train[index])
    max_y_value = np.max(y_train[index])
    maxx = max(abs(max_y_value), abs(min_y_value))
    plt.ylim(-maxx - 0.1 * maxx, maxx + 0.1 * maxx)
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.title(f'Actual {index} Spectra')
    plt.legend()
    plt.savefig(os.path.join('train_cu_vis', f'CD_{index}.png'))
    plt.close()
'''
'''
sns.set(style="white", palette="Set2")
for index in range(0, 600):
    plt.figure(figsize=(10, 6))
    plt.rcParams['axes.linewidth'] = 5
    sns.lineplot(x=range(len(y_test[index])), y=y_test[index], label='Actual Spectrum', color='blue', linewidth=5, linestyle='--')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.title(f'Actual {index} Spectra')
    plt.legend()
    plt.savefig(os.path.join('test_cu_vis', f'CD_{index}.png'))
    plt.close()
'''
# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train * 1e15)
y_test = y_scaler.transform(y_test * 1e15)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)  # Assuming y_train is a matrix of continuous values

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)  # Assuming y_test is a matrix of continuous values

# Define datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
test_dataset = TensorDataset(X_test_tensor.unsqueeze(1), y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Instantiate the model
# model = CDSpectrumCNN()
pretrained_model = CDSpectrumCNN()
pretrained_model.load_state_dict(torch.load('best_model_r2.pth'))
pretrained_model.eval()


for param in pretrained_model.parameters():
    param.requires_grad = False


pretrained_model.fc1 = nn.Linear(1024, 2048, bias=True)
pretrained_model.fc2 = nn.Linear(2048, 120, bias=True)

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.0001)


# Train the model
num_epochs = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)

train_losses = []
test_losses = []

# Define lists to store train and test R2 scores
train_r2_scores = []
test_r2_scores = []
best_test_r2 = -float('inf')
best_test_mae = float('inf')
best_test_mse = float('inf')
best_epoch = -1
for epoch in range(num_epochs):
    pretrained_model.train()
    train_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = pretrained_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)

    pretrained_model.eval()
    test_loss = 0
    predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = pretrained_model(inputs)
            loss = criterion(outputs, labels.to(outputs.device))
            test_loss += loss.item() * inputs.size(0)
            #print(inputs.size(0))
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions)

    # Calculate test loss and R2 score
    test_loss /= len(test_loader.dataset)
    test_r2 = r2_score(y_test, predictions)
    test_mae = mean_absolute_error(y_test, predictions)
    if test_mae < best_test_mae:
        best_test_mae = test_mae
        best_epoch_mae = epoch
        torch.save(pretrained_model.state_dict(), 'best_model_mae_cu.pth')

    if test_loss < best_test_mse:
        best_test_mse = test_loss
        best_epoch_mse = epoch
        torch.save(pretrained_model.state_dict(), 'best_model_mse_cu.pth')

    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_epoch_r2 = epoch
        torch.save(pretrained_model.state_dict(), 'best_model_r2_cu.pth')

    # Append values to lists
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    # train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    if epoch % 50 == 0:
        print(
            f"Epoch {epoch + 1}/{num_epochs},\t\t Train Loss: {train_loss},\t\t Test Loss: {test_loss},\t\t Test R2: {test_r2},\t\t Test MAE: {test_mae}")

print(f"Best mae achieved at epoch {best_epoch_mae + 1}")
print(f"Best r2 achieved at epoch {best_epoch_r2 + 1}")
print(f"Best mse achieved at epoch {best_epoch_mse + 1}")
print(f"Best test mae {best_test_mae}")
print(f"Best test r2 {best_test_r2}")
print(f"Best test mse {best_test_mse}")

# Evaluate the model
# model = CDSpectrumCNN()  # 这里假设你已经定义了 CDSpectrumCNN 模型
pretrained_model.load_state_dict(torch.load('best_model_r2_cu.pth'))
pretrained_model.eval()

# 在测试集上进行预测
test_loss = 0
predictions = []
for inputs, labels in test_loader:
    inputs = inputs.to(device)
    with torch.no_grad():  # 不需要在推断时计算梯度
        outputs = pretrained_model(inputs)
    predictions.append(outputs.cpu().numpy())

# 将预测结果拼接成一个单独的数组
predictions = np.concatenate(predictions)
# Calculate R2 score
r2 = r2_score(y_test, predictions)
print("R2 score:", r2)
y_test = y_scaler.inverse_transform(y_test)
y_test = y_test / 1e15
predictions = y_scaler.inverse_transform(predictions)
predictions = predictions / 1e15

sns.set(style="white", palette="Set2")
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.rcParams['axes.linewidth'] = 5

plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', color='red', linewidth=5)
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss', color='blue', linewidth=5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Losses')
plt.legend()
ax = plt.gca()
ax.xaxis.set_ticks_position('bottom')  # 将刻度放置在上方
ax.yaxis.set_ticks_position('left')  # 将刻度放置在右侧

ax.xaxis.set_tick_params(width=5)  # 设置刻度宽度
ax.yaxis.set_tick_params(width=5)
plt.savefig('Loss_CNN_cu.png')
plt.show()

predictions = np.apply_along_axis(lambda row: savgol_filter(row, window_length, polyorder), axis=1, arr=predictions)
sns.set(style="white", palette="Set2")
x_sequence = np.linspace(400, 800, 120)

for index in range(0, 200):
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.rcParams['axes.linewidth'] = 5
    #x=range(len(y_test[index]))
    #sns.lineplot(x=range(len(y_test[index])), y=y_test[index], label='Actual Spectrum', linewidth=5)
    sns.lineplot(x=x_sequence, y=y_test[index], label='Ground Truth', linestyle='-', linewidth=5, color='red')
    # x=range(len(predictions[index]))
    # 绘制预测光谱
    # sns.lineplot(x=range(len(predictions[index])), y=predictions[index], label='Prediction', linestyle='--',
    # linewidth=4)
    sns.lineplot(x=x_sequence, y=predictions[index], label='Predicted Spectrum', linestyle='--', linewidth=5,
                 color='blue')
    min_y_value = np.min(y_test[index])
    max_y_value = np.max(y_test[index])
    maxx = max(abs(max_y_value), abs(min_y_value))
    plt.ylim(-maxx - 0.1 * maxx, maxx + 0.1 * maxx)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Ext,Diff')
    plt.title(f'Comparison of Actual and Predicted {index} Spectra')
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')  # 将刻度放置在上方
    ax.yaxis.set_ticks_position('left')  # 将刻度放置在右侧

    ax.xaxis.set_tick_params(width=5)  # 设置刻度宽度
    ax.yaxis.set_tick_params(width=5)
    plt.savefig(os.path.join('prediction-wjs-cu', f'CD_{index}.png'))
    plt.close()

'''
for index in range(0, 68):
    plt.plot(range(len(y_test[index])), y_test[index], label='Actual Spectrum')
    plt.plot(range(len(predictions[index])), predictions[index], label='Predicted Spectrum')
    plt.xlabel('Wavelength Index')
    plt.ylabel('Intensity')
    plt.title(f'Comparison of Actual and Predicted {index} Spectra')
    plt.legend()
    plt.savefig(os.path.join('prediction-wjs-test', f'CD {index}.png'))
    plt.close()
'''
print(predictions)