import pandas as pd
import numpy as np

from sklearn import linear_model # import thư viện sklearn.linear_model.LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

data_2 = pd.read_csv("data_2.csv")
print(data_2)


X = data_2.loc[:, ['0']].as_matrix() # Sử dụng "cot dau tien" làm biến giải thích
Y = data_2['0.1'].as_matrix() # Sử dụng "cot thu hai" làm biến mục đích

sk = linear_model.LinearRegression()

# Tạo model suy đoán
sk.fit(X, Y)

# Hệ số hồi quy
print("Hệ số hồi quy: a =", sk.coef_)

# Sai số
print("Sai số: b = ", sk.intercept_)

print("-----------------------------------------")

print("====> Y =", sk.coef_,"X +", sk.intercept_)
print("-----------------------------------------")

# Score
print("Score:", sk.score(X, Y))
print("-----------------------------------------")

# Biểu diễn sự phân bố tập dữ liệu input
# c: color
plt.figure(1)
plt.scatter(X, Y, c='b')

# Đường thẳng hồi quy
plt.plot(X, sk.predict(X))
#plt.show()

# Split the data into training/testing sets
X_train = X[:-100]
X_test = X[-100:]

# Split the targets into training/testing sets
y_train = Y[:-100]
y_test = Y[-100:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs

plt.figure(2)
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()




