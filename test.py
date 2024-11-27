import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Vẽ biểu đồ dữ liệu
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

# Chuẩn bị dữ liệu cho Linear Regression
x = np.hstack((np.ones((N, 1)), x))
w = np.array([0., 1.]).reshape(-1, 1)
numOfIteration = 100
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.000001

# Thuật toán Gradient Descent
for i in range(1, numOfIteration):
    r = np.dot(x, w) - y
    cost[i] = 0.5 * np.sum(r * r)
    w[0] -= learning_rate * np.sum(r)
    w[1] -= learning_rate * np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))

    print(cost[i])

# Dự đoán giá trị
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]), (predict[0], predict[N-1]), 'r')
plt.show()

# Dự đoán giá nhà cho diện tích 50m^2
x1 = 50
y1 = w[0] + w[1] * x1
print('Giá nhà cho 50m^2 là : ', y1)

# Lưu trọng số w vào file .npy
np.save('weight.npy', w)

# Đọc lại trọng số từ file .npy
w = np.load('weight.npy')

# Linear Regression với thư viện sklearn
from sklearn.linear_model import LinearRegression

# Đọc lại dữ liệu
data = pd.read_csv('data_linear.csv').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Vẽ biểu đồ dữ liệu
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

# Huấn luyện mô hình
lrg = LinearRegression()
lrg.fit(x, y)

# Dự đoán giá trị
y_pred = lrg.predict(x)
plt.plot((x[0], x[-1]), (y_pred[0], y_pred[-1]), 'r')
plt.show()

# Lưu nhiều tham số vào file .npz
np.savez('w2.npz', a=lrg.intercept_, b=lrg.coef_)

# Đọc lại tham số từ file .npz
k = np.load('w2.npz')
lrg.intercept_ = k['a']
lrg.coef_ = k['b']
