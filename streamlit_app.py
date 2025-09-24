import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="HW1-1: Interactive Linear Regression Visualizer")
st.title("HW1-1: Interactive Linear Regression Visualizer")

# Sidebar configuration
st.sidebar.header("Configuration")
n = st.sidebar.slider("Number of data point (n)", min_value=20, max_value=500, value=100, step=1)
a = st.sidebar.slider("Coefficient 'a' (y = ax + b + noise)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
var = st.sidebar.slider("Noise Variance (var)", min_value=1.0, max_value=100.0, value=30.0, step=1.0)

# Generate data
np.random.seed(42)
x = np.random.uniform(20, 200, n)
noise = np.random.normal(0, var, n)
y = a * x + 10 + noise

# Add five outliers
outlier_indices = np.random.choice(n, 5, replace=False)
y[outlier_indices] += np.random.normal(200, 50, 5)

# Fit linear regression
X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
fig, ax = plt.subplots()
ax.scatter(x, y, label="Data", alpha=0.5)

ax.scatter(x[outlier_indices], y[outlier_indices], color='red', label="Outliers", s=80)
ax.plot(x, y_pred, color='green', label="Regression Line")
# 在每個 outlier 點旁加上其數值標註
for i, idx in enumerate(outlier_indices):
	ax.text(x[idx], y[idx], f'Outlier{i+1}\n({x[idx]:.1f}, {y[idx]:.1f})', color='red', fontsize=10, fontweight='bold', ha='left', va='bottom')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Regression with Five Outliers')
ax.legend()
st.pyplot(fig)

# Show regression info
st.write(f"**斜率（a）:** {model.coef_[0]:.2f}")
st.write(f"**截距（b）:** {model.intercept_:.2f}")
st.write(f"**R² 分數:** {model.score(X, y):.2f}")
