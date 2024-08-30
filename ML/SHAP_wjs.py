'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
'''
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from shapash.explainer.smart_explainer import SmartExplainer
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")
'''
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''

train_data = pd.read_csv('data/train_g_0603.csv')
test_data = pd.read_csv('data/test_g_0603.csv')
columns = ['pitch', 'fiber_radius', 'n_turns', 'helix_radius', 'total_length', 'curl', 'angle', 'height',
           'total_fiber_length', 'V', 'mass', 'x_y', 'direction']
X_train = train_data[
    ['pitch', 'fiber_radius', 'n_turns', 'helix_radius', 'total_length', 'curl', 'angle', 'height',
     'total_fiber_length', 'V', 'mass', 'x_y', 'direction']]
#, 'x_y', 'direction', 'LCP_RCP', 'section, 'wavelength''
y_train = train_data['g_factor'].values

X_test = test_data[
    ['pitch', 'fiber_radius', 'n_turns', 'helix_radius', 'total_length', 'curl', 'angle', 'height',
     'total_fiber_length', 'V', 'mass', 'x_y', 'direction']]
y_test = test_data['g_factor'].values


print(X_train)
print(y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.fit_transform(y_test.reshape(-1, 1))
'''
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义模型、损失函数和优化器
input_size = X_train.shape[1]
hidden_size1 = 1000
hidden_size2 = 500
model = MLPRegressor(input_size, hidden_size1, hidden_size2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # 在每个epoch结束后计算测试集上的 R^2 值
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        r2 = r2_score(y_test_tensor.numpy(), test_outputs.numpy())
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item()}, Test R^2: {r2}')

# 使用模型进行预测
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
y_pred = y_pred_tensor.numpy()

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


'''



# 定义优化目标函数
def objective(trial):
    # 定义搜索空间
    params = {
        'metric': 'r2',
        'random_state': 2002,
        'n_estimators': trial.suggest_int('n_estimators', 500, 10000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.1),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'num_leaves' : trial.suggest_int('num_leaves', 100, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'cat_smooth' : trial.suggest_int('cat_smooth', 100, 300),
        #'device':'gpu',
        'verbose' : -1
    }

    # 划分训练集和验证集

    # 创建并训练模型
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # 计算验证集上的R^2，并返回其负值（因为Optuna默认最小化目标函数）
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return r2


# 创建Optuna优化对象
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 获取最佳参数
best_params = study.best_params
print("Best Params:", best_params)

# 使用最佳参数训练最终模型
best_model = LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)


joblib.dump(best_model, 'best_model_lgbm_g.pkl')

# Load the model from disk
loaded_model = joblib.load('best_model_lgbm_g.pkl')

# 使用加载的模型进行预测
best_model = loaded_model

# 使用最佳模型进行预测
y_pred = best_model.predict(X_test)

# 评估模型
r2 = r2_score(y_test, y_pred)
print('R^2 Score:', r2)

X_test = pd.DataFrame(X_test, columns=columns)
y_test = pd.DataFrame(y_test, columns=['g_factor'])
y_pred = pd.DataFrame(y_pred, columns=['g_factor'])
explainer  = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test,show=False)
plt.savefig('shap_summary_g.png')

xpl = SmartExplainer(model=best_model)

xpl.compile(
    contributions=shap_values,
    y_pred=y_pred,
    y_target=y_test,
    x=X_test
)
xpl.save('xpl_g.pkl')

#xpl.plot.top_interactions_plot(nb_top_interactions=10)

app = xpl.run_app(title_story='Helix Chiral', port=8080)
