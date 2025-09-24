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
print(data.describe())

plt.scatter(data['Size'], data['Price'], alpha=0.5)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 設定隨機種子以便重現
np.random.seed(42)

# 生成 1000 筆資料
size = np.random.uniform(20, 200, 1000)  # 房屋面積
price = size * 3 + np.random.normal(0, 30, 1000)  # 價格 = 面積 * 3 + 雜訊

# 建立 DataFrame
data = pd.DataFrame({'Size': size, 'Price': price})
print(data.describe())

# 資料視覺化
plt.scatter(data['Size'], data['Price'], alpha=0.5)
plt.xlabel('Size (m²)')
plt.ylabel('Price (10k TWD)')
plt.title('房屋面積與價格散佈圖')
plt.show()

# 資料分割
X = data[['Size']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建模
model = LinearRegression()
model.fit(X_train, y_train)
print(f"斜率（係數）: {model.coef_[0]:.2f}")
print(f"截距: {model.intercept_:.2f}")

# 評估
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

# 部署
new_size = pd.DataFrame({'Size': [85, 150, 60]})
predicted_price = model.predict(new_size)
for s, p in zip(new_size['Size'], predicted_price):
    print(f"預測 {s:.1f} 平方公尺房屋的價格為: {p:.2f} 萬元")