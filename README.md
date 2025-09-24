# 房價預測專案說明

## 1. 業務理解（Business Understanding）
我們的目標是預測房屋價格，這次使用 1000 筆模擬資料。此模型可協助房地產業者快速估算房屋價值，並制定更精準的定價策略。

## 2. 資料理解（Data Understanding）
我們使用 NumPy 生成模擬資料。假設房屋面積在 20 到 200 平方公尺之間，價格與面積呈線性關係，並加入隨機雜訊以模擬真實世界波動。
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定隨機種子以便重現
np.random.seed(42)

# 生成 1000 筆資料
size = np.random.uniform(20, 200, 1000)  # 房屋面積
price = size * 3 + np.random.normal(0, 30, 1000)  # 價格 = 面積 * 3 + 雜訊

# 建立 DataFrame
data = pd.DataFrame({'Size': size, 'Price': price})
```
# 資料初探
print(data.describe())

# 視覺化資料分布

python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定隨機種子以便重現
np.random.seed(42)

# 生成 1000 筆資料
size = np.random.uniform(20, 200, 1000)  # 房屋面積
price = size * 3 + np.random.normal(0, 30, 1000)  # 價格 = 面積 * 3 + 雜訊

# 建立 DataFrame
data = pd.DataFrame({'Size': size, 'Price': price})

```python
print(data.describe())

plt.scatter(data['Size'], data['Price'], alpha=0.5)
plt.xlabel('Size (m²)')
plt.ylabel('Price (10k TWD)')
plt.title('房屋面積與價格散佈圖')
plt.show()

## 3. 資料準備（Data Preparation）
plt.scatter(data['Size'], data['Price'], alpha=0.5)
plt.xlabel('Size (m²)')
plt.ylabel('Price (10k TWD)')
plt.title('房屋面積與價格散佈圖')
plt.show()
我們將資料分成訓練集與測試集，並準備好特徵與標籤：
```python
from sklearn.model_selection import train_test_split

X = data[['Size']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## 4. 建模（Modeling）
使用線性迴歸模型來擬合資料：

```python

python
from sklearn.model_selection import train_test_split

X = data[['Size']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# 模型參數
print(f"斜率（係數）: {model.coef_[0]:.2f}")
print(f"截距: {model.intercept_:.2f}")
```

## 5. 評估（Evaluation）
我們使用 R² 分數與均方誤差來評估模型表現：

```python

python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# 模型參數
print(f"斜率（係數）: {model.coef_[0]:.2f}")
print(f"截距: {model.intercept_:.2f}")
y_pred = model.predict(X_test)

print(f"R² 分數: {r2_score(y_test, y_pred):.2f}")
print(f"均方誤差 (MSE): {mean_squared_error(y_test, y_pred):.2f}")

# 預測 vs 實際視覺化

python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

print(f"R² 分數: {r2_score(y_test, y_pred):.2f}")
print(f"均方誤差 (MSE): {mean_squared_error(y_test, y_pred):.2f}")

plt.scatter(X_test, y_test, label='實際值', alpha=0.5)
plt.scatter(X_test, y_pred, label='預測值', alpha=0.5)
plt.legend()
plt.xlabel('Size (m²)')
plt.ylabel('Price (10k TWD)')
plt.title('預測與實際價格比較')
plt.show()

## 6. 部署（Deployment）
plt.legend()
plt.xlabel('Size (m²)')
plt.ylabel('Price (10k TWD)')
plt.title('預測與實際價格比較')
plt.show()
我們可以用模型來預測任意房屋面積的價格：
```python
new_size = pd.DataFrame({'Size': [85, 150, 60]})
predicted_price = model.predict(new_size)

for s, p in zip(new_size['Size'], predicted_price):
    print(f"預測 {s:.1f} 平方公尺房屋的價格為: {p:.2f} 萬元")
```

python
new_size = pd.DataFrame({'Size': [85, 150, 60]})
predicted_price = model.predict(new_size)

for s, p in zip(new_size['Size'], predicted_price):
    print(f"預測 {s:.1f} 平方公尺房屋的價格為: {p:.2f} 萬元")